#!/usr/bin/env python
# -*- coding: UTF-8 -*- 
# @Time       : 2018/12/17 17:08 
# @Author     : beforechrist221 
# @FileName   : HelloWorld.py 
# @ProjectName: CUU 
import tensorflow as tf
a = tf.constant(5)
b = tf.constant(6)
c = a+b
d = tf.constant(8)
e = c*d

with tf.Session() as sess:
    writer = tf.summary.FileWriter(r'logs/', sess.graph)
    writer.close()
    print(sess.run(c))
    print(sess.run(e))

