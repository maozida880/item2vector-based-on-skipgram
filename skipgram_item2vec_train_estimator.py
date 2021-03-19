#!E:\software\anaconda\envs\maozida880\ python
# _*_ coding: utf-8 _*_
# @Time : 2021/3/10 17:13
# @Author :  maozida880
# @Version：V 0.1
# @File : skipgram_item2vec_train_estimator.py
# @desc :

import tensorflow as tf
import math
import numpy as np

class SkipGram:

    def __init__(self):
        self.data_index = 0
        self.lr = 0.01
        self.batch_size = 100  # 每次迭代训练选取的样本数目
        self.embedding_size = 128  # 生成词向量的维度
        self.window_size = 5  # 考虑前后几个词，窗口大小, skipgram中的中心词-上下文pairs数目就是windowsize *2
        self.num_sampled = 300  # 负样本采样.
        self.num_steps = 20000  # 定义最大迭代次数，创建并设置默认的session，开始实际训练
        self.vocabulary_size = 600
        self.scheme = "all"  # all, window两种

        self.__params__()  # 参数初始化

    def __params__(self):
        self.PARAMS = {}
        self.PARAMS['vocab_size'] = 600
        self.PARAMS['embed_dim'] = self.embedding_size
        self.PARAMS['n_sampled'] = self.num_sampled
        self.PARAMS['batch_size'] = self.batch_size
        self.PARAMS['n_epochs'] = 1

    def __parse_slice(self, *args):
        feature = tf.data.Dataset.from_tensor_slices(*args)
        return feature

    def __parse_batch_for_tabledataset(self, *args):

        feature = tf.strings.to_number(tf.string_split([*args], "|,").values, out_type=tf.int32)  # [None, 1]
        feature = tf.reshape(feature, name='feature',shape=[-1, 2])  # [None, 2]
        return feature

    def __parse2(self,x):
        input = x[:,0]
        label = tf.reshape(x[:, 1], [-1,1])
        return input, label

    def train_input_fn_from_odps(self, batch_size=1024,epoch=2):
        dataset = tf.data.TextLineDataset('item2vec_test_data.txt')
        dataset = dataset.map(self.__parse_batch_for_tabledataset)
        dataset = dataset.flat_map(self.__parse_slice)
        dataset = dataset.shuffle(buffer_size=1000).batch(batch_size=batch_size).repeat(epoch).prefetch(100)
        dataset = dataset.map(lambda x: self.__parse2(x))
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        init_op = iterator.initializer
        return dataset

    def model_fn(self, features, labels , mode, params):

        PARAMS = params
        lr = PARAMS['lr']

        nce_weights = tf.get_variable(name='softmax_W', shape=[PARAMS['vocab_size'], PARAMS['embed_dim']])
        nce_biases = tf.get_variable(name='softmax_b', shape=[PARAMS['vocab_size']])
        with device("cpu"):
            embeddings = tf.Variable(tf.random_uniform(shape=[ PARAMS['vocab_size'],  PARAMS['embed_dim'] ], minval=-1.0, maxval=1.0))
            embed = tf.nn.embedding_lookup(embeddings, features)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # 定义loss，损失函数，tf.reduce_mean求平均值，# 得到NCE损失(负采样得到的损失)
            loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,  # 权重
                                                 biases=nce_biases,  # 偏差
                                                 labels=labels,  # 输入的标签
                                                 inputs=embed,  # 输入向量
                                                 num_sampled=PARAMS['n_sampled'],  # 负采样的个数
                                                 num_classes=PARAMS['vocab_size']))  # 类别数目

            optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(loss, global_step=tf.train.get_global_step())
            # 定义优化器，使用梯度下降优化算法
            # optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=optimizer)

        if mode == tf.estimator.ModeKeys.PREDICT:
            # 计算每个词向量的模，并进行单位归一化，保留词向量维度
            normalized_E = tf.nn.l2_normalize(embeddings, -1)
            sample_E = tf.nn.embedding_lookup(normalized_E, features)
            similarity = tf.matmul(sample_E, normalized_E, transpose_b=True)

            return tf.estimator.EstimatorSpec(mode, predictions=similarity)

    def train(self):
        # 先做训练，在考虑保存模型，再考虑并行化

        estimator = tf.estimator.Estimator(
            self.model_fn,
            params={
                'vocab_size': self.PARAMS['vocab_size'],
                'embed_dim': self.PARAMS['embed_dim'],
                'n_sampled': self.PARAMS['n_sampled'],
                'lr': self.lr
            }
        )

        estimator.train(
            input_fn= lambda : self.train_input_fn_from_odps(
                self.batch_size,
                epoch=None
            ),
            max_steps=self.num_steps
        )

def main(_):
    model = SkipGram()
    model.train()

if __name__ == "__main__":
    tf.app.run()