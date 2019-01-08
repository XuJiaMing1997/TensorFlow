import numpy as np
import os
import sys
import time
import tensorflow as tf

import keras.backend as K
import keras

from keras.utils.np_utils import to_categorical
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from PIL import Image
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from tensorflow.contrib.slim import nets
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from keras.initializers import RandomNormal


import matplotlib.pyplot as plt



DECAY = 0.95
ifSin = True
# False means square test

def myScheduler(epoch,lr):
    if not ifSin:
        if epoch in [500,600]:
            return lr*DECAY
        elif epoch in [700,800]:
            return lr*(DECAY**2)
        elif epoch == 900:
            return lr*(DECAY**3)
        else:
            return lr
        # return lr
    else:
        if epoch in [500,600,700]:
            return lr*DECAY
        elif epoch in [800,900]:
            return lr*(DECAY**2)
        else:
            return lr
my_LR_decay = keras.callbacks.LearningRateScheduler(myScheduler)

class get_LR(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch%100 == 0:
            print('epoch: {0} present LR: {1}'.format(epoch,K.get_value(self.model.optimizer.lr)))
        return
    def on_train_begin(self, logs=None):
        print('original lr: {0}'.format(K.get_value(self.model.optimizer.lr)))

my_get_LR = get_LR()


# #vilina ---- 10000epoch --time:145s---loss:8e-5-----val_loss:2e-5
# synthesizeX = np.linspace(-5,5,100)
# np.random.shuffle(synthesizeX)
# noise = np.random.normal(0,0.001,size=[100,])
# # noise = np.random.normal(0,0.05,size=[100,])
# synthesizeY = np.square(synthesizeX) + noise
#
# fine grained  -------- loss: 0.0002  val_loss:5e-5 ------------ number_flow: 1e6
synthesizeX = np.linspace(-5,5,1000)
noise = np.random.normal(0,0.001,size=[1000,])
if ifSin:
    synthesizeY = np.sin(synthesizeX) + noise
else:
    synthesizeY = np.square(synthesizeX) + noise

# # repeated ----------- time:45s----loss:0.0001 val_loss: 8.9e-6  ------number flow:1e6
# #                                       0.0002           5e-5
# synthesizeX = np.linspace(-5,5,100)
# synthesizeX = np.tile(synthesizeX,10)
# np.random.shuffle(synthesizeX)
# noise = np.random.normal(0,0.001,size=[1000,])
# synthesizeY = np.square(synthesizeX) + noise

if not ifSin:
    # for square
    validationX = np.linspace(-2,2,100)
    validationY = np.square(validationX)
else:
    validationX = np.linspace(-3,3,100)
    validationY = np.sin(validationX)

# for i,j in zip(synthesizeX,synthesizeY):
#     print(i,j)

myInput = Input(shape=[1,],name='myInput')
# 0.001  original steddev
x = Dense(40,activation='relu',name='Dense_Layer_1',
          kernel_initializer=RandomNormal(mean=0.0, stddev=0.05))(myInput)
x = Dense(20,activation='relu',name='Dense_Layer_2',
          kernel_initializer=RandomNormal(mean=0.0, stddev=0.05))(x)
myOutput = Dense(1,activation=None,name='Output_Layer',
                 kernel_initializer=RandomNormal(mean=0.0, stddev=0.05))(x)

myModel = Model(inputs=myInput,outputs=myOutput,name='LinearApproximation')
# myModel.compile(keras.optimizers.SGD(lr=0.01,momentum=0.9),loss='mean_absolute_error')
myModel.compile(keras.optimizers.Adam(),loss='mean_squared_error')

myModel.summary()

for i in  range(2):
    time.sleep(2)
    print('time wait: {0}'.format(2*(i+1)))

tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs',histogram_freq=1,write_graph=True,write_images=True)

time_start = time.clock()
his = myModel.fit(synthesizeX,synthesizeY,batch_size=50,epochs=1000,verbose=0,
                  validation_data=[validationX,validationY],callbacks=[my_get_LR,tbCallBack,my_LR_decay])
time_end = time.clock()
print('time elapsed: {0}'.format(time_end-time_start))

for i,j in zip(his.epoch,his.history['loss']):
    if i%100 == 0:
        print('epoch: {0} loss: {1}'.format(i,j))
for i,j in zip(his.epoch,his.history['val_loss']):
    if i%70 == 0:
        print('epoch: {0} val_loss: {1}'.format(i,j))

####
print('===========Stage Two==========')
####
if not ifSin:
    myModel.compile(keras.optimizers.SGD(lr=0.0001,momentum=0,nesterov=False),loss='mean_absolute_error')
else:
    myModel.compile(keras.optimizers.SGD(lr=0.001,momentum=0,nesterov=False),loss='mean_squared_error')

tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs',histogram_freq=1,write_graph=True,write_images=True)

if not ifSin:
    his = myModel.fit(synthesizeX,synthesizeY,batch_size=50,epochs=1000,verbose=0,
                      validation_data=[validationX,validationY],callbacks=[tbCallBack,my_LR_decay])
else:
    his = myModel.fit(synthesizeX,synthesizeY,batch_size=50,epochs=1000,verbose=0,
                      validation_data=[validationX,validationY],callbacks=[tbCallBack,my_LR_decay])

for i,j in zip(his.epoch,his.history['loss']):
    if i%100 == 0:
        print('epoch: {0} loss: {1}'.format(i,j))
for i,j in zip(his.epoch,his.history['val_loss']):
    if i%70 == 0:
        print('epoch: {0} val_loss: {1}'.format(i,j))

####
print('===========test===========')
testX = np.array([0,1,2,3,4,5,1.5,-1,-2,-3,-1.5])
if not ifSin:
    testY = np.square(testX)
else:
    testY = np.sin(testX)
res = myModel.predict(testX).flatten()
print('X = {0}'.format(testX))
print('Y = {0}'.format(testY))
print('Predict = {0}'.format(res.astype(float)))
print('Loss = ')
for i,j in zip(res,testY):
    print('{:f}'.format(abs(i-j)))
###

showX = np.linspace(-8,8,1000)
if not ifSin:
    showY = np.square(showX)
else:
    showY = np.sin(showX)
show_res = myModel.predict(showX).flatten()

plt.figure()

l1, = plt.plot(showX,show_res,color='red',linestyle='--',linewidth=0.5,label='predict')
l2, = plt.plot(showX,showY,color='green',linewidth=0.5,label='real')
plt.scatter(synthesizeX,synthesizeY)
plt.legend(handles=[l1,l2],labels=['predict','real'],loc='best')

plt.show()



