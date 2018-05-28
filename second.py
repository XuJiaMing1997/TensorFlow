import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf

def addLayer(input,input_size,output_size,activation_function = None):
    with tf.name_scope('B_Layer'):
        with tf.name_scope('B_Weight'):
            Weights = tf.Variable(tf.random_normal([input_size,output_size]))
        with tf.name_scope('B_biases'):
            biases = tf.Variable(tf.zeros([1,output_size]) +  0.1)
        with tf.name_scope('B_Combination'):
            Combination = tf.matmul(input,Weights) + biases#interesting addition operation
        if not activation_function:
            return Combination
        else:
            return activation_function(Combination)

#Synthesize raw data
sX = np.linspace(-1,1,500)[:,np.newaxis]
Noise = np.random.normal(0,0.05,sX.shape)
sY = np.square(sX) + Noise

#NN input
with tf.name_scope('B_input'):
    Xinput = tf.placeholder(tf.float32,[None,1],name = 'Xinput')
    Yinput = tf.placeholder(tf.float32,[None,1],name = 'Yinput')

#Build NN construction
L1 = addLayer(Xinput,1,10,activation_function = tf.nn.relu)
Prediction = addLayer(L1,10,1,None)

#Optimmization part
with tf.name_scope('B_loss'):
    loss = tf.reduce_mean(tf.square(Prediction - Yinput))
with tf.name_scope('B_train'):
    train_step = tf.train.GradientDescentOptimizer(0.02).minimize(loss)


with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)













