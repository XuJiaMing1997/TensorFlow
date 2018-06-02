import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

#import data
mnist = input_data.read_data_sets('MNIST_data',one_hot=False)

#Visualize parameter
train_epoch = 10
batch_size = 256
image_size = 784#28*28 = 784
lr = 0.01
show_examples = 10

#structure parameter
Layer1_size = 512
Layer2_size = 256
Layer3_size = 128
Layer4_size = 64

#input
inputX = tf.placeholder(tf.float32,[None,image_size])

#parameter initialization
Weights_encoder = {
    'Layer1':tf.Variable(tf.random_normal([image_size,Layer1_size],stddev=0.1)),
    'Layer2':tf.Variable(tf.random_normal([Layer1_size,Layer2_size],stddev=0.1)),
    'Layer3':tf.Variable(tf.random_normal([Layer2_size,Layer3_size],stddev=0.1)),
    'Layer4':tf.Variable(tf.random_normal([Layer3_size,Layer4_size],stddev=0.1))
}
Weights_decoder = {
    'Layer1':tf.Variable(tf.random_normal([Layer4_size,Layer3_size],stddev=0.1)),
    'Layer2':tf.Variable(tf.random_normal([Layer3_size,Layer2_size],stddev=0.1)),
    'Layer3':tf.Variable(tf.random_normal([Layer2_size,Layer1_size],stddev=0.1)),
    'Layer4':tf.Variable(tf.random_normal([Layer1_size,image_size],stddev=0.1))
}

biases_encoder = {
    'Layer1':tf.Variable(tf.random_normal([Layer1_size],stddev=0.1)),
    'Layer2':tf.Variable(tf.random_normal([Layer2_size],stddev=0.1)),
    'Layer3':tf.Variable(tf.random_normal([Layer3_size],stddev=0.1)),
    'Layer4':tf.Variable(tf.random_normal([Layer4_size],stddev=0.1))
}

# Note: this part is not as fully mirror reflection as the imput_biases,which is same as Weights_decoder axis2
biases_decoder = {
    'Layer1':tf.Variable(tf.random_normal([Layer3_size],stddev=0.1)),
    'Layer2':tf.Variable(tf.random_normal([Layer2_size],stddev=0.1)),
    'Layer3':tf.Variable(tf.random_normal([Layer1_size],stddev=0.1)),
    'Layer4':tf.Variable(tf.random_normal([image_size],stddev=0.1))
}

# structure
# Encoder Layer
EL1 = tf.nn.sigmoid(tf.matmul(inputX,Weights_encoder['Layer1']) + biases_encoder['Layer1'])
EL2 = tf.nn.sigmoid(tf.matmul(EL1,Weights_encoder['Layer2']) + biases_encoder['Layer2'])
EL3 = tf.nn.sigmoid(tf.matmul(EL2,Weights_encoder['Layer3']) + biases_encoder['Layer3'])
EL4 = tf.nn.sigmoid(tf.matmul(EL3,Weights_encoder['Layer4']) + biases_encoder['Layer4'])

#Decoder Layer
DL1 = tf.nn.sigmoid(tf.matmul(EL4,Weights_decoder['Layer1']) + biases_decoder['Layer1'])
DL2 = tf.nn.sigmoid(tf.matmul(DL1,Weights_decoder['Layer2']) + biases_decoder['Layer2'])
DL3 = tf.nn.sigmoid(tf.matmul(DL2,Weights_decoder['Layer3']) + biases_decoder['Layer3'])
Prediction = tf.nn.sigmoid(tf.matmul(DL3,Weights_decoder['Layer4']) + biases_decoder['Layer4'])

#loss
loss = tf.reduce_mean(tf.square(inputX - Prediction))

#train
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(train_epoch):
        batch_loop = mnist.train.num_examples//batch_size
        for _ in range(batch_loop):
            batchX, batchY = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={inputX:batchX})
        print("Present eoch {0}\nloss: {1}".format(epoch,sess.run(loss,feed_dict={inputX:batchX})))
    print("Optimization stage finished!")
    #plot
    test_show = sess.run(Prediction,feed_dict={inputX:mnist.test.images[20:20+show_examples]})
    f,a = plt.subplots(2,10,figsize=(10,2))
    for _ in range(show_examples):
        a[0][_].imshow(np.reshape(mnist.test.images[20+_],[28,28]))
        a[1][_].imshow(np.reshape(test_show[_],[28,28]))
    plt.show()
    plt.pause(10)
























