import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def addLayer(input,input_size,output_size,name,activation_function = None):
    LayerName = 'Layer{s}'.format(s=name)
    with tf.name_scope('B_Layer'):
        with tf.name_scope('B_Weight'):
            Weights = tf.Variable(tf.random_normal([input_size,output_size]))
            tf.summary.histogram(LayerName + '/Weight',Weights)
        with tf.name_scope('B_biases'):
            biases = tf.Variable(tf.zeros([1,output_size]) +  0.1)
            tf.summary.histogram(LayerName + '/biases',biases)
        with tf.name_scope('B_Combination'):
            Combination = tf.matmul(input,Weights) + biases#interesting addition operation
        tf.summary.histogram(LayerName + '/output',Combination)
        if not activation_function:
            return Combination
        else:
            return activation_function(Combination)

#input interface
Xinput = tf.placeholder(tf.float32,[None,784])
Yinput = tf.placeholder(tf.float32,[None,10])

#train data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#structure
Prediction = addLayer(Xinput,784,10,1,activation_function = tf.nn.softmax)

#loss
cross_entropy =   tf.reduce_mean(-tf.reduce_sum(Yinput * tf.log(Prediction),axis = 1))

#train
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(1000):
        batchX,batchY = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={Xinput:batchX,Yinput:batchY})
        if i%50 == 0:
            testX = mnist.test.images
            testY = mnist.test.labels
            Ypredicted = sess.run(Prediction,feed_dict={Xinput:testX,Yinput:testY})
            BoolAccuracy = tf.equal(tf.argmax(Ypredicted,axis = 1),tf.argmax(testY,1))
            Accuracy = tf.reduce_mean(tf.cast(BoolAccuracy,tf.float32))
            print(sess.run(Accuracy,feed_dict={Xinput:testX,Yinput:testY}))














