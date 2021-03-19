#!E:\software\anaconda\envs\maozida880\python
# _*_ coding: utf-8 _*_
# @Time : 2021/3/17 17:26
# @Author :  maozida880
# @Version：V 0.1
# @File : skipgrm_item2vec_distribution.py.py
# @desc :



import os

os.environ['VAR_PARTITION_THRESHOLD'] = '262144'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.python.training import queue_runner_impl
import math
import numpy as np
import json


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("tables","table1,table2", "tables info")
tf.app.flags.DEFINE_string("osstables", "本地文件", "oss file")

tf.app.flags.DEFINE_integer("max_train_step", 100000, "max train step")
tf.app.flags.DEFINE_integer("save_summary_steps", 500, "save summary pre 10000 step")
tf.app.flags.DEFINE_integer("save_checkpoint_and_eval_step", 500, "save checkpoint and eval step")
tf.app.flags.DEFINE_string("checkpoint_dir",
                           '保存路径',
                           "model checkpoint dir permanet part")

tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, eval, eval_train_loss}")

tf.app.flags.DEFINE_string("job_name", "", "job name")
tf.app.flags.DEFINE_integer("task_index", None, "Worker or server index")
tf.app.flags.DEFINE_string("ps_hosts", "", "ps hosts")
tf.app.flags.DEFINE_string("worker_hosts", "", "worker hosts")
tf.app.flags.DEFINE_integer("inter_op_parallelism_threads", 12, "inter_op_parallelism_threads")
tf.app.flags.DEFINE_integer("intra_op_parallelism_threads", 12, "intra_op_parallelism_threads")


class ThreadStartHook(tf.train.SessionRunHook):
    def __init__(self, is_chief, sync_init_op):
        self.is_chief = is_chief
        self.sync_init_op = sync_init_op

    def after_create_session(self, session, coord):
        self.coord = coord
        print("queue runner started")
        if self.is_chief:
            if self.sync_init_op is not None:
                self.sync_init_op.run()
        self.threads = tf.train.start_queue_runners(coord=coord, sess=session)

    def end(self, session):
        self.coord.request_stop()
        if self.is_chief:
            self.coord.join(self.threads)

class SkipGram:

    def __init__(self):
        self.data_index = 0
        self.lr = 0.0001
        self.batch_size = 4096  # 每次迭代训练选取的样本数目
        self.embedding_size = 32  # 生成词向量的维度
        self.window_size = 5  # 考虑前后几个词，窗口大小, skipgram中的中心词-上下文pairs数目就是windowsize *2
        self.num_sampled = 100  # 负样本采样.
        self.num_steps = FLAGS.max_train_step  # 定义最大迭代次数，创建并设置默认的session，开始实际训练
        self.scheme = "all"  # all, window两种

        self.save_checkpoint_and_eval_step = FLAGS.save_checkpoint_and_eval_step
        self.max_train_step = FLAGS.max_train_step
        self.save_summary_steps = FLAGS.save_summary_steps

        self.inter_op_parallelism_threads = FLAGS.inter_op_parallelism_threads
        self.intra_op_parallelism_threads = FLAGS.intra_op_parallelism_threads

        self.__params__()  # 参数初始化
        self.__path_init__()  # 文件解析
        self.__init_distribute__()  # 分布式配置

    def __params__(self):
        self.PARAMS = {}
        self.PARAMS['vocab_size'] = 6032331
        self.PARAMS['embed_dim'] = self.embedding_size
        self.PARAMS['n_sampled'] = self.num_sampled
        self.PARAMS['batch_size'] = self.batch_size
        self.PARAMS['n_epochs'] = 1

    def __path_init__(self):
        self.tables = FLAGS.tables
        tables_list = self.tables.split(",")
        self.pair_table = tables_list[0]
        self.val_pair_table = tables_list[1]

        self.osstables = FLAGS.osstables
        tables_list = self.osstables.split(",")
        self.oss_pair_table = tables_list[0]

        self.checkpoint_path = FLAGS.checkpoint_dir

    def __init_distribute__(self):
        self.every_n_steps = 100
        self.task_index = FLAGS.task_index

        ps_hosts = FLAGS.ps_hosts.split(",")
        worker_hosts = FLAGS.worker_hosts.split(",")

        print("task type is {}".format(FLAGS.task_type))
        if 'TF_CONFIG' in os.environ:
            print("tfconfig is {}".format(os.environ['TF_CONFIG']))
        # 2
        if FLAGS.task_type == 'predict':
            self.worker_num = len(worker_hosts)
            print("in prediction")
            if 'TF_CONFIG' in os.environ:
                print("tfconfig is {}".format(os.environ['TF_CONFIG']))

        else:  # train
            print('old task_index: ' + str(FLAGS.task_index))
            if FLAGS.task_index > 1:
                self.task_index = FLAGS.task_index - 1
            print('new task_index: ' + str(self.task_index))
            self.worker_num = len(worker_hosts) - 1

			
            if len(worker_hosts):
                cluster = {"chief": [worker_hosts[0]], "ps": ps_hosts, "worker": worker_hosts[2:]}
                if FLAGS.job_name == "ps":
                    os.environ['TF_CONFIG'] = json.dumps(
                            {'cluster': cluster, 'task': {'type': FLAGS.job_name, 'index': FLAGS.task_index}})
                elif FLAGS.job_name == "worker":
                    if FLAGS.task_index == 0:
                        os.environ['TF_CONFIG'] = json.dumps(
                                {'cluster': cluster, 'task': {'type': "chief", 'index': 0}})
                    elif FLAGS.task_index == 1:
                        os.environ['TF_CONFIG'] = json.dumps(
                                {'cluster': cluster, 'task': {'type': "evaluator", 'index': 0}})
                    else:
                        os.environ['TF_CONFIG'] = json.dumps(
                                {'cluster': cluster, 'task': {'type': FLAGS.job_name, 'index': FLAGS.task_index - 2}})
        if 'TF_CONFIG' in os.environ:
            print("after changed tfconfig is {}".format(os.environ['TF_CONFIG']))

        self.is_chief = False
        if self.task_index == 0 and FLAGS.job_name == "worker":
            print("This is chief")
            self.is_chief = True

        self.hook_sync_replicas = None
        self.sync_init_op = None


    def __parse_slice(self, *args):
        feature = tf.data.Dataset.from_tensor_slices(args)
        return feature

    def __parse_batch_for_tabledataset(self, *args):
        feature = tf.strings.to_number(tf.string_split([*args], "|,").values, out_type=tf.int32)  # [None, 1]
        feature = tf.reshape(feature, name='feature',shape=[-1, 2])  # [None, 2]
        return feature

    def __parse2(self,x):
        input = x[:,0]
        label = tf.reshape(x[:, 1], [-1,1])
        return input, label

    def train_input_fn_from_odps(self, pair_table, batch_size=1024,epoch=2, slice_id=0, slice_count=1):
        with tf.device('/cpu:0'):
            dataset = tf.data.TableRecordDataset([pair_table],
                                                 record_defaults=[''],
                                                 slice_count=slice_count,
                                                 slice_id=slice_id,
                                                 num_threads=1,
                                                 capacity=1)
            dataset = dataset.map(self.__parse_batch_for_tabledataset)
            dataset = dataset.flat_map(self.__parse_slice)
            dataset = dataset.shuffle(buffer_size=1000).batch(batch_size=batch_size).repeat(epoch).prefetch(100)
            dataset = dataset.map(lambda x: self.__parse2(x))
            return dataset

    def val_input_fn_from_odps(self, pair_table, batch_size=1024,epoch=2, slice_id=0, slice_count=1):
        with tf.device('/cpu:0'):
            dataset = tf.data.TableRecordDataset([pair_table],
                                                 record_defaults=[''],
                                                 slice_count=slice_count,
                                                 slice_id=slice_id,
                                                 num_threads=1,
                                                 capacity=1)
            dataset = dataset.map(self.__parse_batch_for_tabledataset)
            dataset = dataset.flat_map(self.__parse_slice)
            dataset = dataset.shuffle(buffer_size=4096*10).batch(batch_size=batch_size).repeat(epoch).prefetch(100)
            dataset = dataset.map(lambda x: self.__parse2(x))
            return dataset

    def model_fn(self, features, labels , mode, params):

        PARAMS = params
        lr = PARAMS['lr']

        nce_weights = tf.get_variable(name='softmax_W', shape=[PARAMS['vocab_size'], PARAMS['embed_dim']])
        nce_biases = tf.get_variable(name='softmax_b', shape=[PARAMS['vocab_size']])
        # E = tf.get_variable('embedding', [PARAMS['vocab_size'], PARAMS['embed_dim']])
        # embed = tf.nn.embedding_lookup(E, features)  # forward activation
        with tf.device('/cpu:0'):
            embeddings = tf.Variable(tf.random_uniform(shape=[ PARAMS['vocab_size'],  PARAMS['embed_dim'] ], minval=-1.0, maxval=1.0))
            embed = tf.nn.embedding_lookup(embeddings, features)
        

        # 定义loss，损失函数，tf.reduce_mean求平均值，# 得到NCE损失(负采样得到的损失)
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, 
        									 biases=nce_biases, 
        									 labels=labels, 
        									 inputs=embed,  
        									 num_sampled=PARAMS['n_sampled'], 
        									 num_classes=PARAMS['vocab_size'])) 

        # 定义优化器，使用Adam梯度下降优化算法
        #optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)

        optimizer = tf.train.GradientDescentOptimizer(1.0)

        optimizer = tf.train.SyncReplicasOptimizer(optimizer,
        									 replicas_to_aggregate=self.worker_num,
        									 total_num_replicas=self.worker_num,
        									 use_locking=False)



        if mode == tf.estimator.ModeKeys.PREDICT:
            # 计算每个词向量的模，并进行单位归一化，保留词向量维度
            # norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            # normalized_embeddings = embeddings / norm

            normalized_E = tf.nn.l2_normalize(embeddings, -1)
            sample_E = tf.nn.embedding_lookup(normalized_E, features)
            similarity = tf.matmul(sample_E, normalized_E, transpose_b=True)

            return tf.estimator.EstimatorSpec(mode, loss = loss, predictions=similarity)
		
        print("hook_sync_replicas is set")
        self.hook_sync_replicas = optimizer.make_session_run_hook(is_chief=self.is_chief, num_tokens=0)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        chief_queue_runner = optimizer.get_chief_queue_runner()
        queue_runner_impl.add_queue_runner(chief_queue_runner)
        self.sync_init_op = optimizer.get_init_tokens_op()
        
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,
                                              training_hooks=[self.hook_sync_replicas])

    def train(self):
        # 先做训练，在考虑保存模型，再考虑并行化

        print("......................Start training......................")
        session_config = tf.ConfigProto()
        session_config.intra_op_parallelism_threads = self.intra_op_parallelism_threads
        session_config.inter_op_parallelism_threads = self.inter_op_parallelism_threads

        estimator = tf.estimator.Estimator(
            self.model_fn,
            params={
                'vocab_size': self.PARAMS['vocab_size'],
                'embed_dim': self.PARAMS['embed_dim'],
                'n_sampled': self.PARAMS['n_sampled'],
                'lr': self.lr
            },
            config = tf.estimator.RunConfig(
            session_config=session_config,
            model_dir=self.checkpoint_path,
            tf_random_seed=2020,
            save_summary_steps=self.save_summary_steps,
            save_checkpoints_steps=self.save_checkpoint_and_eval_step,
            keep_checkpoint_max=1000)
        )



        hooks = []
        if self.is_chief:
            # Hook that counts steps per second.
            hook_counter = tf.train.StepCounterHook(output_dir=self.checkpoint_path, every_n_steps=self.every_n_steps)
            hooks.append(hook_counter)

        train_spec = tf.estimator.TrainSpec(
            input_fn= lambda : self.train_input_fn_from_odps(
                self.pair_table,
                self.batch_size,
                epoch=None,
                slice_id = self.task_index,
                slice_count = self.worker_num
            ),
            max_steps=self.num_steps,
            hooks=hooks
        )

        eval_spec =  tf.estimator.EvalSpec(
            input_fn= lambda : self.val_input_fn_from_odps(
                self.val_pair_table,
                self.batch_size,
                epoch=None,
                slice_id = 0,
                slice_count = 1
            ),
            steps=None,
            start_delay_secs=60,
            throttle_secs=60
        )

        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

        print("end")

def main(_):
    model = SkipGram()
    model.train()

if __name__ == "__main__":
    tf.app.run()
