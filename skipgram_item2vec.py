#!/usr/bin/env python3
# coding: utf-8
# File: (app)item2vector.py
# Author: maozida880 
# Date: 21-01-18

import collections
import math
import random
import numpy as np
import tensorflow as tf
import common_io
import random
import queue
import time
import threading

tf.app.flags.DEFINE_string("tables", "初处理输入表,训练输入表 ", "初处理输入表：是一个用户的item index列表, [1 2 3 4 5]；训练输入表：是初处理表变为pair对的表")
tf.app.flags.DEFINE_string("outputs", "初处理输出表", "初处理输出表就是训练输入表")

tf.app.flags.DEFINE_string("checkpointDir", "模型保存地址", "output info")
tf.app.flags.DEFINE_string("pairname", "训练输入表", "用于训练的输入表")
tf.app.flags.DEFINE_string("modelpath",'模型保存地址', "model，与checkpointDir一样，通道不同而已")

tf.app.flags.DEFINE_integer("lowfrequency", 100, "lowfrequency limit")

FLAGS = tf.app.flags.FLAGS

class SkipGram:
    def __init__(self):
        self.data_index = 0

        self.tables = FLAGS.tables
        self.output = FLAGS.outputs
        # self.modelpath = FLAGS.checkpointDir

        # input config
        self.__input_init__()

        self.min_count = FLAGS.lowfrequency # 最低词频，保留模型中的词表  lee这个要都保留
        self.batch_size = 2000 # 每次迭代训练选取的样本数目
        self.embedding_size = 128  # 生成词向量的维度
        self.window_size = 5  # 考虑前后几个词，窗口大小, skipgram中的中心词-上下文pairs数目就是windowsize *2
        self.num_sampled = 300  # 负样本采样.
        self.num_steps = 200000 # 定义最大迭代次数，创建并设置默认的session，开始实际训练
        self.vocabulary_size = 1000000
        # self.words = self.read_data(self.dataset)  # lee 词的表达都用整体表示
        self.scheme = "all"  # all, window两种

        self.pair_name = FLAGS.pairname

        # make pair
        self.__pair_init__()
        
        #read data
        # self.pair_list = self.readfile2list()
        # self.length_pair_list = len(self.pair_list)

        self.pair_list = []
        self.queue_preline = queue.Queue(maxsize=1)
        self.queue_readingstate = queue.Queue(maxsize=1)

    def __input_init__(self):
        tables_list = self.tables.split(",")
        self.input_table = tables_list[0]
        self.pair_table = tables_list[1]
        self.output_table = self.output
        self.modelpath = FLAGS.checkpointDir.split(",")[1]
        self.output_path = FLAGS.checkpointDir.split(",")[0]
        self.pair_file = self.output
        self.dataset = self.input_fn()  # lee 以句子为表示的列表

    def __pair_init__(self):
        #make data
        print("2. build dataset")
        # self.data, _, _, _ = self.build_dataset(self.words, self.min_count)
        self.data = self.dataset
        print("dataset is {}".format(self.dataset[0]))
        print( "words count is {}".format(len(self.dataset)) )
        self.pair_list = self.centerword_label_pair_generate(self.window_size,self.data,self.scheme)

    #define the input
    def input_fn(self):

        print("1. reading data and make dictionary")
        dataset = []
        with common_io.table.TableReader(self.input_table) as f1:
            cnt = f1.get_row_count()
            cnt = 4800000
            print("all need to read %d lines"%cnt)
            for i in range(cnt):
                line = f1.read(1)
                index_list = line[0][1].split(",")
                dataset.append(index_list)
                if i % 500000 == 0: print("reading: %d"%i)
        return dataset

    # 定义读取数据的函数，并把数据转成列表
    def read_data(self, dataset):
        print("将句子列表融合成大列表")
        words = []
        for data in dataset:
            words.extend(data)
        return words
    #创建数据集
    def build_dataset(self, words, min_count):
        # 创建词汇表，过滤低频次词语，这里使用的人是mincount>=5，其余单词认定为Unknown,编号为0,
        # 这一步在gensim提供的wordvector中，采用的是minicount的方法
        #对原words列表中的单词使用字典中的ID进行编号，即将单词转换成整数，储存在data列表中，同时对UNK进行计数
        count = [['UNK', -1]]   #频率词表
        count.extend([item for item in collections.Counter(words).most_common() if item[1] >= min_count])
        dictionary = dict()   # 序号词表  {word:index}
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()  #lee 词的编号
        unk_count = 0
        for word in words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count += 1
            # data.append(index)
        count[0][1] = unk_count
        # 将dictionary中的数据反转，即可以通过ID找到对应的单词，保存在reversed_dictionary中
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        #lee 构建句子的词
        for sentence in self.dataset:
            sentence_index_list = []
            for word in list(set(sentence)):
                if word in dictionary.keys():
                    index = dictionary[word]
                else:
                    index = 0
                sentence_index_list.append(index)
            data.append(sentence_index_list)

        return data, count, dictionary, reverse_dictionary

    #这个模块是产生 pair list的，产生[(centre,label)]
    def centerword_label_pair_generate(self,window_size,data,scheme = "all"):
        #问题，要不要让一个句子的中心词每次只产生10个（input/label）对，分成sentence_word_num//10组，将组的顺序打乱？
        # pair_list = [] # (input, label)  lee  把所有的标签都放进来，这个数量是sentence_word_num^2 * sentence_num (10^2)^2*10^4

        # lee 格式是[[(pair1),(pair2),...(pairN)]sentence1, []sentence2,...,[]sentenceN]
        print("2.1 center word label pair generate")
        lasttime = time.time()
        tmp_list = []
        # with tf.gfile.GFile(self.output_path + self.pair_name, 'w') as fw:
        with common_io.table.TableWriter(self.output_table) as fw:

            print("2.2 all data count is %d"%len(data))
            for num,sentence in enumerate(data):
                random.shuffle(sentence)  #乱序
                sentence_pair_list = []
                sentence_len = len(sentence)
                for i in range(sentence_len):
                    center_word = sentence[i]
                    count = 0
                    for j in range(sentence_len):

                        random_sampling = random.randint(0, sentence_len-1)
                        if random_sampling != i:
                            label_word = sentence[random_sampling]
                            sentence_pair_list.append((center_word,label_word))
                        if count>20:break;
                        count+=1

                if scheme == "all":
                    #方案1: 是中心词全量连续窗口方案
                    # pair_list.extend(sentence_pair_list)
                    pass
                elif scheme == "window":
                    #方案2: 顺序打乱方案： 句子内以窗口大小的2倍为单位，进行乱序排位计算
                    label_num = 2*window_size  # 一个前后窗口的标签数目
                    if sentence_len<label_num+1:continue;  #如果窗口不够用，不算了
                    sentence_pair_num = len(sentence_pair_list)   #pair数量
                    sentence_group_num = sentence_pair_num//label_num #组数量
                    last_num = sentence_pair_num%label_num   #余数

                    #共分为 sentence_group_num+1 组
                    sentence_pair_list_by_group = [sentence_pair_list[num*label_num:(num+1)*label_num] for num in range(sentence_group_num)]
                    sentence_pair_list_by_group.append(sentence_pair_list[-1-last_num:-1])

                    random.shuffle(sentence_pair_list_by_group)

                    sentence_pair_list_shuffle = []
                    for ss in sentence_pair_list_by_group:
                        sentence_pair_list_shuffle.extend(ss)
                    sentence_pair_list = sentence_pair_list_shuffle
                    #pair_list.extend(sentence_pair_list_shuffle)
                else:
                    print("input error!")
                    break
                if len(sentence_pair_list):
                    for pair in sentence_pair_list:
                        # tmp_list.append((str(pair[0])+","+str(pair[1])))
                        tmp_list.append( str(pair[0])+","+str(pair[1]) )

                if num % 30000 == 0:
                    start = lasttime
                    middletime = time.time()

                if num %100 == 0:
                    if num>1:
                        # fw.write("\n".join(tmp_list)+"\n")
                        ready_to_write_list = ["|".join(tmp_list)]
                        fw.write( ready_to_write_list,range(1) )
                        tmp_list = []

                if num%30000 == 0:
                    lasttime = time.time()
                    print("line num is {}, every 30000 imei use time is {}, write 30000 time is {}, process time is {}".format(
                        num,lasttime-start,lasttime-middletime, middletime-start))



        # return pair_list

    def readfile2list(self):
        print("3. reading data and make dictionary")
        dataset = []
        with common_io.table.TableReader(self.pair_table) as f1:
            cnt = f1.get_row_count()
            print("all need to read %d lines"%cnt)
            for i in range(cnt):
                line = f1.read(1)
                center_word = line[0][0]
                label_word = line[0][1]
                dataset.append((center_word, label_word))
                if i % 500000 == 0: print("reading: %d"%i)
        return dataset

    def linepara(self, line):
        data_list = []
        line_pair_list = line[0][0].split("|")

        for pair in line_pair_list:
            pair_list = pair.split(",")
            pair_tuple = (pair_list[0], pair_list[1])
            data_list.append(pair_tuple)
        return data_list

    def lineproduce(self):
        # 提供数据的线程
        # 1. 解析表；2. 满足batch需求； 3. 提供线程服务  4. 生产者，通过self.queue_readingstate传状态停止
        queue_preline = self.queue_preline
        queue_readingstate = self.queue_readingstate

        reader = common_io.table.TableReader(
            self.pair_table
        )
        total_records_num = reader.get_row_count()
        print("start to read odps table and need read %d lines"%total_records_num)
        count = 0
        for _ in range(total_records_num):
            data=reader.read(1)
            data_list = self.linepara(data)
            queue_preline.put(data_list)  # 生产者装置数据
            print("=========produce %d line======="%count)
            count += 1
            StateFlag = queue_readingstate.get()  # 获取状态
            if StateFlag:  #传状态，控制停止

                # 存储
                print("embedding size is %d" % (len(self.final_embeddings)))
                print("writing embedding")
                final_embeddings = self.final_embeddings
                print(final_embeddings[0])

                print("save model path is %s" % (self.modelpath + "model"))
                fw = tf.gfile.GFile(self.modelpath + "model", 'w')

                for index, item in enumerate(final_embeddings):
                    if index % 50000 == 0:
                        print("save dictionary %d lines" % index)
                    # fw.write(reverse_dictionary[index] + '\t' + ','.join([str(vec) for vec in item]) + '\n')
                    fw.write(str(index) + '\t' + ','.join([str(vec) for vec in item]) + '\n')
                fw.close()

                print("可能还有语料，但是Step已经到了上限")
                break

    def __readdata(self):
        # 什么时候取队列中的数据：在batch将num消耗掉
        self.data_index = 0
        queue_preline = self.queue_preline
        self.pair_list = queue_preline.get()
        self.length_pair_list = len(self.pair_list)
        print("consumer get queue data, line number is %d"%(len(self.pair_list)))

    #生成训练样本，assert断言：申明其布尔值必须为真的判定，如果发生异常，就表示为假
    def generate_batch(self, batch_size, window_size):
        # 该函数根据训练样本中词的顺序抽取形成训练集
        # 这个函数的功能是对数据data中的每个单词，分别与前一个单词和后一个单词生成一个batch，
        # 即[data[1],data[0]]和[data[1],data[2]]，其中当前单词data[1]存在batch中，前后单词存在labels中
        # batch_size:每个批次训练多少样本
        # num_skips: 为每个单词生成多少样本（本次实验是2个），batch_size必须是num_skips的整数倍,这样可以确保由一个目标词汇生成的样本在同一个批次中。
        # window_size:单词最远可以联系的距离（本次实验设为1，即目标单词只能和相邻的两个单词生成样本），2*window_size>=num_skips
        '''
        eg:
        input_batch, labels = generate_batch(batch_size = 8, num_skips = 2, window_size = 1)
        #Sample data [0, 5241, 3082, 12, 6, 195, 2, 3137, 46, 59] ['UNK', 'anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used']
        #假设取num_steps为2, window_size为1, batchsize为8
        #input_batch:[5242, 3084, 12, 6]
        #labels[0, 3082, 5241, 12, 3082, 6, 12, 195]
        print(input_batch)  [5242 5242 3084 3084   12   12    6    6]，共8维
        print(labels) [[   0] [3082] [  12] [5242] [   6] [3082] [  12] [ 195]]，共8维
        '''

        input_batch = np.ndarray(shape = (batch_size), dtype = np.int32) #建一个batch大小的数组，保存任意单词
        labels = np.ndarray(shape = (batch_size, 1), dtype = np.int32)#建一个（input_batch，1）大小的二维数组，保存任意单词前一个或者后一个单词，从而形成一个pair

        label_num = 2*window_size  # 一个前后窗口的标签数目
        buffer = collections.deque(maxlen = batch_size) #建立一个结构为双向队列的缓冲区，大小不超过3，实际上是为了构造bath以及labels，采用队列的思想

        # self.pair_list要改成可以迭代的
        # print("generate batch data")
        for _ in range(batch_size):
            buffer.append(self.pair_list[self.data_index])       #lee 先装1个字  问题：能否灵活装配？
            self.data_index = (self.data_index + 1) % self.length_pair_list  #lee 由于self.data_index是全域的，他的更新是全程的
            # 即使更新到尾部，也可以与头部连接上
        # print(buffer)
        for i in range(batch_size):  #lee 这意味着batch是同时进行多少个窗口

            input_batch[i] = int(buffer[i][0])
            labels[i, 0] = int(buffer[i][1])
            buffer.append(self.pair_list[self.data_index])
            self.data_index = (self.data_index + 1) % self.length_pair_list

        return input_batch, labels


    def train_wordvec(self, vocabulary_size, batch_size, embedding_size, window_size, num_sampled, num_steps):
        #定义Skip-Gram Word2Vec模型的网络结构
        graph = tf.Graph()
        with graph.as_default():
            #输入数据， 大小为一个batch_size
            train_inputs = tf.placeholder(tf.int32, shape = [batch_size])
            #目标数据，大小为[batch_size]
            train_labels = tf.placeholder(tf.int32, shape = [batch_size, 1])

            #使用cpu进行训练  哥有GPU
            # with tf.device('/cpu:0'):

            #生成一个vocabulary_size×embedding_size的随机矩阵，为词表中的每个词，随机生成一个embedding size维度大小的向量，
            #词向量矩阵，初始时为均匀随机正态分布，tf.random_uniform((4, 4), minval=low,maxval=high,dtype=tf.float32)))
            #随机初始化一个值于介于-1和1之间的随机数，矩阵大小为词表大小乘以词向量维度
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            #tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。用于查找对应的wordembedding， ，将输入序列向量化
            #tf.nn.embedding_lookup(params, ids, partition_strategy='mod', name=None, validate_indices=True, max_norm=None)
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            #全连接层，Wx+b,设置W大小为，embedding_size×vocabulary_size的权重矩阵，模型内部参数矩阵，初始为截断正太分布
            nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev = 1.0 / math.sqrt(embedding_size)))
            # 全连接层，Wx+b,设置W大小为，vocabulary_size×1的偏置
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


            #定义loss，损失函数，tf.reduce_mean求平均值，# 得到NCE损失(负采样得到的损失)
            loss = tf.reduce_mean(tf.nn.nce_loss(weights = nce_weights,# 权重
                                                biases = nce_biases,# 偏差
                                                labels = train_labels,# 输入的标签
                                                inputs = embed, # 输入向量
                                                num_sampled = num_sampled,# 负采样的个数
                                                num_classes = vocabulary_size))# 类别数目
            #定义优化器，使用梯度下降优化算法
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
            #计算每个词向量的模，并进行单位归一化，保留词向量维度
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims = True))
            normalized_embeddings = embeddings / norm
            #初始化模型变量
            init = tf.global_variables_initializer()

        #基于构造网络进行训练

        queue_readingstate = self.queue_readingstate
        # queue_readingstate.put(False)
        with tf.Session(graph = graph) as session:
            #初始化运行
            init.run()
            #定义平均损失
            average_loss = 0
            #每步进行迭代

            data_offer_steps = 0  # 数据一共提供了多少步
            print("start to training")
            for num, step in enumerate(range(num_steps)):
                # print("train num is %d, train step is %d"%(num,step))
                if len(self.pair_list) == 0:
                    queue_readingstate.put(False)
                    self.__readdata()
                    max_steps_pre_line = len(self.pair_list)//batch_size
                    print("get data %d"%max_steps_pre_line)
                    assert max_steps_pre_line > 0
                    data_offer_steps += max_steps_pre_line

                if num >= data_offer_steps:
                    queue_readingstate.put(False)
                    self.__readdata()
                    max_steps_pre_line = len(self.pair_list) // batch_size
                    assert max_steps_pre_line > 0
                    data_offer_steps += max_steps_pre_line

                batch_inputs, batch_labels = self.generate_batch(batch_size, window_size)
                #feed_dict是一个字典，在字典中需要给出每一个用到的占位符的取值。
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
                #计算每次迭代中的loss
                _, loss_val = session.run([optimizer, loss], feed_dict = feed_dict)
                #计算总loss
                average_loss += loss_val
                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    print("Average loss at step ", step, ":", average_loss)
                    average_loss = 0
                if num%1000==0:print("train num is %d"%num)



            print("training is over")
            self.final_embeddings = normalized_embeddings.eval()
            # 线程停止命令符：队列传递停止命令，在所有tensor结束后
            queue_readingstate.put(True)





        # return final_embeddings
    #保存embedding文件
    def save_embedding(self, final_embeddings=None, reverse_dictionary=None):
        # final_embeddings = self.final_embeddings
        fw = tf.Gfile.open(self.modelpath+"model", 'w')
        for index, item in enumerate(final_embeddings):
            # fw.write(reverse_dictionary[index] + '\t' + ','.join([str(vec) for vec in item]) + '\n')
            fw.write(str(index) + '\t' + ','.join([str(vec) for vec in item]) + '\n')
        fw.close()
    #训练主函数
    def train(self):
        # data, count, dictionary, reverse_dictionary = self.build_dataset(self.words, self.min_count)
        # vocabulary_size = len(count)
        vocabulary_size = self.vocabulary_size
        self.train_wordvec(vocabulary_size, self.batch_size, self.embedding_size, self.window_size, self.num_sampled, self.num_steps)
        final_embeddings = self.final_embeddings
        self.save_embedding(final_embeddings)

    def run(self):
        vocabulary_size = self.vocabulary_size
        print("start muli threding")
        print("producter")
        Thread_reading = threading.Thread(target=self.lineproduce, args=())
        print("consumer")
        Thread_train = threading.Thread(target=self.train_wordvec, args=(vocabulary_size, self.batch_size, self.embedding_size, self.window_size, self.num_sampled, self.num_steps))
        Thread_train.setDaemon(True)

        Thread_reading.start()
        Thread_train.start()

        # print("writing embedding")
        # final_embeddings = self.final_embeddings
        # self.save_embedding(final_embeddings)




def main():

    vector = SkipGram()
    # vector.run()



if __name__ == '__main__':
    main()
