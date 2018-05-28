
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf

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

#Synthesize raw data
sX = np.linspace(-1,1,500)[:,np.newaxis]
Noise = np.random.normal(0,0.05,sX.shape)
sY = np.square(sX) + Noise

#NN input
with tf.name_scope('B_input'):
    Xinput = tf.placeholder(tf.float32,[None,1],name = 'Xinput')
    Yinput = tf.placeholder(tf.float32,[None,1],name = 'Yinput')

#Build NN construction
L1 = addLayer(Xinput,1,10,name=1,activation_function = tf.nn.relu)
Prediction = addLayer(L1,10,1,2,None)

#Optimmization part
with tf.name_scope('B_loss'):
    loss = tf.reduce_mean(tf.square(Prediction - Yinput))
    tf.summary.scalar('OverallCost/loss',loss)
with tf.name_scope('B_train'):
    train_step = tf.train.GradientDescentOptimizer(0.02).minimize(loss)


with tf.Session() as sess:
    mergeData = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logs',sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)
    for _ in range(1000):
        sess.run(train_step,feed_dict={Xinput : sX,Yinput : sY})
        if  _ % 50 == 0:
            showData = sess.run(mergeData,feed_dict={Xinput : sX,Yinput : sY})
            writer.add_summary(showData,_)
























