import numpy as np
import os
import sys
import time
import tensorflow as tf

import matplotlib.pyplot as plt



Animation = False
# Animated figure will be automatically closed in PyCharm, so add pause(10000)


synthsizeX = np.linspace(-5,5,1000)[:,np.newaxis]
noise = np.random.normal(0,0.001,size=np.shape(synthsizeX))
realY = 6 * synthsizeX - 5
synthsizeY =  realY + noise


validationX = np.linspace(1,10,1000)[:,np.newaxis]
validationY = 6 * validationX - 5


W = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

inputX = tf.placeholder(dtype=tf.float32,shape=[None,1])
inputY = tf.placeholder(dtype=tf.float32,shape=[None,1])

Predict = tf.matmul(inputX,W) + b

loss = tf.reduce_mean(tf.square(Predict-inputY))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

if Animation:
    fig = plt.figure()
    subfig = fig.add_subplot(1,1,1)
    subfig.scatter(synthsizeX,synthsizeY)
    plt.ion()
    plt.show()
    predictX = np.linspace(-10,10,1000)[:,np.newaxis]
    predictY = 6 * predictX - 5
    subfig.plot(predictX,predictY,'g-',lw=1.5)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        for j in range(20):
            _ = sess.run(train_op,feed_dict={inputX:synthsizeX[50*j:50*(j+1)],inputY:synthsizeY[50*j:50*(j+1)]})
        if i%10 == 0:
            recv_loss = sess.run(loss,feed_dict={inputX:synthsizeX,inputY:synthsizeY})
            val_loss = sess.run(loss,feed_dict={inputX:validationX,inputY:validationY})
            print('epoch: {0} loss: {1} val_loss: {2}'.format(i*10,recv_loss,val_loss))
        if Animation:
            if i%100 == 0:
                try:
                    subfig.lines.remove(line[0])
                except Exception:
                    pass
                recv_pred = sess.run(Predict,feed_dict={inputX:predictX})
                line = subfig.plot(predictX,recv_pred,'r--',lw = 1.5)
                plt.pause(0.1)

    print('Real_W = 6 Real_b = -5\nW = {0} b = {1}'.format(sess.run(W),sess.run(b)))

    if Animation:
        plt.pause(10000)

    if not Animation:
        predictX = np.linspace(-10,10,1000)[:,np.newaxis]
        predictY = 6 * predictX - 5
        recv_predict = sess.run(Predict,feed_dict={inputX:predictX})

        plt.figure()
        l1, =plt.plot(predictX,recv_predict,color='red',linestyle='--',linewidth=1.5)
        l2, = plt.plot(predictX,predictY,color = 'green',linewidth=1.5)
        plt.scatter(synthsizeX,synthsizeY)
        plt.legend(handles=[l1,l2],labels=['predict','real'],loc='best')
        plt.show()


