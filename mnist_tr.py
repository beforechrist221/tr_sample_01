# -*- coding:utf-8 -*-
"""
使用全连接神经网络，基于mnist数据集训练手写数字识别模型
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class Mnist(object):
    def __init__(self):
        pass

    def __get_weights__(self, input_size, output_size):
        """
        根据用户输入的input_size和output_size初始化网络权值
        :param input_size: 输入大小
        :param output_size: 输出大小
        :return: 未经初始化的网络权值，是一个tensor
        """
        weights = tf.get_variable("weights",
                                  [input_size, output_size],
                                  dtype="float",
                                  initializer=tf.initializers.random_normal)
        return weights

    def __get_biases__(self, input_size):
        """
        根据用户输入的Input_size初始化网络偏置
        :param input_size:输入大小
        :return:未经初始化的网络偏置，是一个tensor
        """
        biases = tf.get_variable("biases",
                                 [input_size],
                                 dtype="float",
                                 initializer=tf.initializers.constant)
        return biases

    def add_layer(self, input, input_size, output_size, activation=None):
        """
        根据用户的输入创建网络层
        :param input:当前网络层次的输入
        :param input_size:网络输入的大小，比如第一层的网络输入为784
        :param output_size:经过当前网络处理之后的输出，比如1024
        :param activation:当前网络的激活函数，比如sigmoid
        :return:activation(input*weights+biases)
        """
        weights = self.__get_weights__(input_size, output_size)
        biases = self.__get_biases__(output_size)
        layer = tf.matmul(input, weights) + biases
        if activation:
            layer = activation(layer)
        return layer

    def __inference__(self, input):
        """
        构建网络的正向传播
        :param input:网络的输入
        :return:
        """
        with tf.variable_scope("first_layer"):
            first_layer = self.add_layer(input, 784, 1024, tf.nn.sigmoid)

        with tf.variable_scope("second_layer"):
            second_layer = self.add_layer(first_layer, 1024, 10, tf.nn.sigmoid)

        return second_layer

    def train_model(self, input):
        """
        训练网络
        :param input:网络输入
        :return:None
        """
        x = tf.placeholder("float", shape=[None, 784], name="features")
        y = tf.placeholder("float", shape=[None, 10], name="targets")
        output = self.__inference__(x)
        output = tf.nn.softmax(output)
        entropy = -tf.reduce_mean(y*tf.log(output))
        cross_entropy = tf.train.GradientDescentOptimizer(0.00009).minimize(entropy)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter(r'logs/', sess.graph)
            writer.close()
            for _ in range(2):
                _, t = sess.run([cross_entropy, entropy], feed_dict={x:input.train.images,y:input.train.labels})
                print(t)

mnist = Mnist()
data = input_data.read_data_sets("data\\", one_hot=True)
print(data.train.labels)
mnist.train_model(data)
