import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#from tensorflow.examples.tutorials.mnist import input_data

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

#import data
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# load data
digits = load_digits()
X = digits.data
Y = digits.target
Y = LabelBinarizer().fit_transform(Y)
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=.3)

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
        Combination = tf.nn.dropout(Combination,keep)
        if not activation_function:
            return Combination
        else:
            return activation_function(Combination)

#input
with tf.name_scope('Xinput'):
    #Xinput = tf.placeholder(tf.float32,[None,784],name='Xinput')
    Xinput = tf.placeholder(tf.float32,[None,64],name = 'Yinput')
    Yinput = tf.placeholder(tf.float32,[None,10],name = 'Yinput')
    keep = tf.placeholder(tf.float32,name='keep_prob')

#structure
'''
L1 = addLayer(Xinput,784,100,1,activation_function=tf.nn.tanh)
L2 = addLayer(L1,100,10,2,activation_function = tf.nn.softmax)
'''
L1 = addLayer(Xinput,64,50,1,activation_function=tf.nn.tanh)
L2 = addLayer(L1,50,10,2,activation_function=tf.nn.softmax)

#loss
with  tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(Yinput * tf.log(L2),axis = 1))
    tf.summary.scalar('OverallCost',cross_entropy)

#train
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
    mergeData = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('logs/train',sess.graph)
    test_writer = tf.summary.FileWriter('logs/test',sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(1000):
        '''
        batchX,batchY = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={Xinput:batchX,Yinput:batchY})
        '''
        sess.run(train_step,feed_dict={Xinput:trainX,Yinput:trainY,keep : 0.5})
        if i % 50 == 0:
            print(sess.run(cross_entropy,feed_dict={Xinput:trainX,Yinput:trainY,keep : 0.5}))
            '''
            print (sess.run(cross_entropy,feed_dict={Xinput:batchX,Yinput:batchY}))
            testX = mnist.test.images
            testY = mnist.test.labels
            showTrainData = sess.run(mergeData,feed_dict={Xinput:batchX,Yinput:batchY})
            train_writer.add_summary(showTrainData,i)
            showTestData = sess.run(mergeData,feed_dict={Xinput:testX,Yinput:testY})
            test_writer.add_summary(showTestData,i)
            '''
            showTrainData = sess.run(mergeData,feed_dict={Xinput:trainX,Yinput:trainY,keep : 1})
            showTestData = sess.run(mergeData,feed_dict={Xinput:testX,Yinput:testY,keep : 1})
            train_writer.add_summary(showTrainData,i)
            test_writer.add_summary(showTestData,i)







