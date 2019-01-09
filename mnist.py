# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
import numpy as np


def scale_images(img):
    """
    对图片进行处理:
    如果图片大小不为28*28,则将其修改为28*28大小
    对图片中的每个像素进行[0-255]归一化
    :param img: 图片流
    :return: 图片数据，期望序列
    """
    if img.width != 28 or img.height != 28:
        img = img.resize((28, 28))
    images = []
    for i in range(28):
        for j in range(28):
            rgb = img.getpixel((j, i))
            images.append(rgb)
    images = np.array(images)
    images = 1.0 - images / 255.0
    return images


class Mnist(object):
    def __init__(self):
        pass

    def __get_weights__(self, weight_shape, regularization=None):
        """
        get weights by input parameters from user
        :parm input_size: input length
        :para output_size: net work will output
        return : wei jing chu shi hua de tensor
        """
        weights = tf.get_variable(
            "weigths",
            shape=weight_shape,
            dtype="float",
            initializer=tf.initializers.random_normal)
        if regularization:
            loss = regularization(weights)
            tf.add_to_collection("losses", loss)
        return weights

    def __get_biases__(self, output_size):
        """
        根据用户输入的output_size创建网络偏置
        :param output_size: 当前层次网络的节点个数
        return: 未经初始化的tensor
        """
        biases = tf.get_variable(
            "biases",
            shape=[output_size],
            dtype="float",
            initializer=tf.initializers.constant)
        return biases

    def __add_cnn_layer__(self,
                          input,
                          filter_size,
                          output_size,
                          strides=[1, 1, 1, 1],
                          padding="VALID",
                          activation=None):
        """
        新增一个卷积层
        :param input:卷积层的输入
        :param filter_size: 卷积核的大小，是一个四位列表，比如[5,5,1,6]
        :param output_size:输出的大小
        :param strides:滑动距离
        :param padding:边缘
        :param activation:激活函数，比如sigmoid
        :return:
        """
        filter = self.__get_weights__(filter_size,
                                      tf.contrib.layers.l2_regularizer(0.001))
        biases = self.__get_biases__(output_size)
        result = tf.nn.conv2d(input, filter, strides, padding) + biases
        return tf.nn.sigmoid(result)

    def __add_max_pooling_layer__(self,
                                  input,
                                  filter,
                                  strides=[1, 2, 2, 1],
                                  padding="VALID"):
        """
        增加一个池化层
        :param input:输入
        :param filter: 池化核的大小
        :param strides: 滑动距离
        :param padding: 边缘
        :return:
        """
        result = tf.nn.max_pool(input, filter, strides, padding)
        return result

    def __add_nn_layer__(self, input, input_size, output_size,
                         activation=None):
        """
        往网络中添加新的网络层次
        :param input:网络的输入
        :param input_size:网络输入的长度
        :param output_Size:网络的输出个数
        :param activation:函数，将线性结果映射为可分类的结果
        return :经过计算的网络输出, 是一个tensor
        """
        weights = self.__get_weights__([input_size, output_size],
                                       tf.contrib.layers.l2_regularizer(0.001))
        biases = self.__get_biases__(output_size)
        layer = tf.matmul(input, weights) + biases
        if activation:
            layer = activation(layer)
        return layer

    def __inference__(self, input):
        """
        定义神经网络的正向传播
        :param input: 网络输入,是一个[?, 784]的列表
        return: 经过网络正向传播计算后的输出，是一个tensor
        """
        # reshape input
        input = tf.reshape(input, (-1, 28, 28, 1))
        # 添加一个卷积层，大小是5*5*1*6
        with tf.variable_scope("first_cnn", reuse=tf.AUTO_REUSE):
            first = self.__add_cnn_layer__(input, [5, 5, 1, 6], 6)
            first = self.__add_max_pooling_layer__(first, [1, 2, 2, 1])

        # 添加一个卷积层，大小是5*5*6*16
        with tf.variable_scope("second_cnn", reuse=tf.AUTO_REUSE):
            second = self.__add_cnn_layer__(first, [5, 5, 6, 16], 16)
            second = self.__add_max_pooling_layer__(second, [1, 2, 2, 1])

        # 添加一个全连接层，大小是[4*4*16,120]
        second = tf.reshape(second, (-1, 4 * 4 * 16))
        with tf.variable_scope("third_nn", reuse=tf.AUTO_REUSE):
            third = self.__add_nn_layer__(second, 4 * 4 * 16, 120,
                                          tf.nn.sigmoid)

        # 添加一个全连接层，大小是[120, 84]
        with tf.variable_scope("forth_nn", reuse=tf.AUTO_REUSE):
            forth = self.__add_nn_layer__(third, 120, 84, tf.nn.sigmoid)

        # 添加一个输出层，大小是[84, 10]
        with tf.variable_scope("output_nn", reuse=tf.AUTO_REUSE):
            output = self.__add_nn_layer__(forth, 84, 10, tf.nn.sigmoid)

        return output

    def train_mnist(self, input, batch_size=1000):
        # 定义一个placehoder用来存储图片
        x = tf.placeholder("float", shape=[None, 784], name="features")
        # 定义一个placeholder用来存储target
        y = tf.placeholder("float", shape=[None, 10], name="target")

        # 调用正向传播函数，得到正向传播的结果
        output = self.__inference__(x)
        # 调用softmax函数，将结果转换为概率结果，此时结果类似：[0.03, 0.12, 0.38, 0.5]
        after_softmax = tf.nn.softmax(output)
        # 损失函数
        loss = -tf.reduce_mean(y * tf.log(after_softmax))
        # 定义梯度下降过程
        cross_entropy = tf.train.AdamOptimizer(0.00099).minimize(loss)
        # declare accuracy=mean(cast(equals(after_softmax, y),"float"))
        accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(y, 1), tf.argmax(after_softmax, 1)),
                "float"))
        # 创建一个对象，用于保存模型
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # 初始化所有变量
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(r"logs", sess.graph)
            writer.close()
            # declare a variable batches
            batches = input.train.num_examples // batch_size
            if input.train.num_examples % batch_size != 0:
                batches += 1
            for i in range(15):
                # 批量梯度下降
                for _ in range(batches):
                    data = input.train.next_batch(batch_size)
                    sess.run(cross_entropy, feed_dict={x: data[0], y: data[1]})
                acc = sess.run(
                    accuracy,
                    feed_dict={
                        x: input.validation.images,
                        y: input.validation.labels
                    })
                print(acc)
                if i % 10 == 0:
                    saver.save(sess, "models\\mnist.ckpt")

    def evaluate(self, test_data):
        # 定义一个placehoder用来存储图片
        x = tf.placeholder("float", shape=[None, 784], name="features")
        # 定义一个placeholder用来存储target
        y = tf.placeholder("float", shape=[None, 10], name="target")

        output = self.__inference__(x)
        after_softmax = tf.nn.softmax(output)
        accuracy = tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(y, 1), tf.argmax(after_softmax, 1)),
                "float"))
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "models\\mnist.ckpt")
            acc = sess.run(
                accuracy, feed_dict={
                    x: test_data.images,
                    y: test_data.labels
                })
            print(acc)

    def get_pic_number(self, img):
        img = scale_images(img).reshape(-1, 784)
        # 定义一个placehoder用来存储图片
        x = tf.placeholder("float", shape=[None, 784], name="features")
        result = tf.argmax(tf.nn.softmax(self.__inference__(x)), 1)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, "models\\mnist.ckpt")
            acc = sess.run(result, feed_dict={x: img})
            return acc


data = input_data.read_data_sets("data\\", one_hot=True)
mist = Mnist()
mist.train_mnist(data)
