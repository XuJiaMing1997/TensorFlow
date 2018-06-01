import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

#import data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#Hyperparameter
batch_size = 128
hiddenUnit_size = 128
time_steps = 28#mnist image length of column axis
single_inputs = 28#mnist image length of row axis
output_class = 10#one hot encoding
lr = 0.001
iterMax = 10000

#input
inputX = tf.placeholder(tf.float32,[None,time_steps,single_inputs])
inputY = tf.placeholder(tf.float32,[None,10])

#Weight
Weights = {
    'in':tf.Variable(tf.random_normal([single_inputs,hiddenUnit_size])),
    'out':tf.Variable(tf.random_normal([hiddenUnit_size,output_class]))
}

#biases
biases = {
    'in':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[hiddenUnit_size])),
    'out':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[output_class]))
}

#RNN
def RNN(input):
    #preparation
    flatX = tf.reshape(input,[batch_size*time_steps,single_inputs])
    #input layer
    inputMul = tf.matmul(flatX,Weights['in']) + biases['in']
    #[128*28,128]
    inputMul = tf.reshape(inputMul,[-1,time_steps,hiddenUnit_size])
    #[128,28,128]

    #hidden Layer
    cell = tf.nn.rnn_cell.BasicLSTMCell(hiddenUnit_size,forget_bias=1.0,state_is_tuple=True)
    init_state = cell.zero_state(batch_size,dtype=tf.float32)

    output,final_state = tf.nn.dynamic_rnn(cell,inputMul,initial_state=init_state,time_major=False)

    #output layer
    result = tf.matmul(final_state[1],Weights['out']) + biases['out']
    return result

#RNN compute
Prediction = RNN(inputX)

#loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=inputY,logits=Prediction))

#train
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

#compute accuracy
AccuracyBool = tf.equal(tf.argmax(Prediction,1),tf.argmax(inputY,1))
Accuracy = tf.reduce_mean(tf.cast(AccuracyBool,tf.float32))


#fig
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
scatter = []

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for _ in range(iterMax):
        batchX, batchY = mnist.train.next_batch(batch_size)
        batchX = batchX.reshape([batch_size,time_steps,single_inputs])#strange ??can not use tf.reshape
        sess.run(train_step,feed_dict={inputX:batchX,inputY:batchY})
        if _ % 50 == 0:
            scatter.append(sess.run(Accuracy,feed_dict={inputX:batchX,inputY:batchY}))
            print (scatter[-1])
    line = ax.plot(scatter,range(len(scatter)),'r-',lw = 3)
    plt.ion()
    plt.show()






















