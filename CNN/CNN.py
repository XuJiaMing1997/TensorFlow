import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#import data---28 * 28 = 784
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#input
Xinput = tf.placeholder(tf.float32,[None,784])
Yinput = tf.placeholder(tf.float32,[None,10])
NodeKeep = tf.placeholder(tf.float32)
Ximage = tf.reshape(Xinput,[-1,28,28,1])

def addFullConnectedLayer(input,input_size,output_size,activation_function = None):
    Weights = tf.Variable(tf.random_normal([input_size,output_size],stddev=0.1))
    biases = tf.Variable(tf.zeros([1,output_size]) +  0.1)
    Combination = tf.matmul(input,Weights) + biases#interesting addition operation
    #Combination = tf.nn.dropout(Combination,NodeKeep)
    if not activation_function:
        return tf.nn.dropout(Combination,NodeKeep)
    else:
        return tf.nn.dropout(activation_function(Combination),NodeKeep)

def addConvLayer(input,input_hight,output_hight):
    Weights = tf.Variable(tf.truncated_normal([5,5,input_hight,output_hight],stddev=0.1))
    biases = tf.Variable(tf.constant(0.1,shape=[output_hight]))
    ConvRes = tf.nn.relu(tf.nn.conv2d(input,Weights,[1,1,1,1],padding='SAME') + biases)
    PoolRes = tf.nn.max_pool(ConvRes,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    #image resize to half of it
    return PoolRes


#structure
Conv1 = addConvLayer(Ximage,1,32)
Conv2 = addConvLayer(Conv1,32,64)
Conv2_flat = tf.reshape(Conv2,[-1,7*7*64])
FCLayer1 = addFullConnectedLayer(Conv2_flat,7*7*64,1024,tf.nn.relu)

#the normal distribution must be 0.1--stddev or if use default 1--stddev the algorithm will not get convergence!!!!!!
Weights = tf.Variable(tf.random_normal([1024,10],stddev=0.1))
biases = tf.Variable(tf.zeros([1,10]) +  0.1)
Prediction = tf.nn.softmax(tf.matmul(FCLayer1,Weights) + biases)

#loss
cross_entropy =   tf.reduce_mean(-tf.reduce_sum(Yinput * tf.log(Prediction),axis = 1))

#train
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(1000):
        batchX,batchY = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={Xinput:batchX,Yinput:batchY,NodeKeep:0.5})
        if i%50 == 0:
            testX = mnist.test.images[:1000]
            testY = mnist.test.labels[:1000]
            Ypredicted = sess.run(Prediction,feed_dict={Xinput:testX,NodeKeep:1})
            BoolAccuracy = tf.equal(tf.argmax(Ypredicted,axis = 1),tf.argmax(testY,1))
            Accuracy = tf.reduce_mean(tf.cast(BoolAccuracy,tf.float32))
            print(sess.run(Accuracy,feed_dict={Xinput:testX,Yinput:testY,NodeKeep:1}))















