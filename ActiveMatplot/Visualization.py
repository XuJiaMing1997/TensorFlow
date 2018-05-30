import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn
import tensorflow as tf

def addLayer(input,input_size,output_size,activation_function = None):
    Weights = tf.Variable(tf.random_normal([input_size,output_size]))
    biases = tf.Variable(tf.zeros([1,output_size]) +  0.1)
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
Xinput = tf.placeholder(tf.float32,[None,1],name = 'Xinput')
Yinput = tf.placeholder(tf.float32,[None,1],name = 'Yinput')

#Build NN construction
L1 = addLayer(Xinput,1,10,activation_function = tf.nn.relu)
Prediction = addLayer(L1,10,1,None)

#Optimmization part
loss = tf.reduce_mean(tf.square(Prediction - Yinput))
train_step = tf.train.GradientDescentOptimizer(0.02).minimize(loss)

#
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(sX, sY)
plt.ion()
plt.show()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(1000):
        sess.run(train_step,feed_dict={Xinput : sX,Yinput : sY})
        if  _ % 50 == 0:
            Print_value = sess.run(loss,feed_dict={Xinput : sX,Yinput : sY})
            print(Print_value)
            Prediction_print = sess.run(Prediction,feed_dict={Xinput : sX})
            try:
                ax.lines.remove(line[0])
            except Exception:
                pass
            line = ax.plot(sX,Prediction_print,'r-',lw = 5)
            plt.pause(0.1)












