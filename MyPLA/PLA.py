import matplotlib.pyplot as plt
import numpy as np

#Synthesize data
SampleX = np.zeros([100,2])
SampleX[:50,0] = np.random.normal(8,2,50)
SampleX[:50,1] = np.random.normal(10,2,50)
SampleX[50:,0] = np.random.normal(2,2,50)
SampleX[50:,1] = np.random.normal(1,2,50)
SampleY = np.ones([100])
SampleY[50:] *= -1


#Hyperparameter
SampleNum = 100


#Normalization
Xmean = np.mean(SampleX,axis=0)
Xdev = np.std(SampleX,axis = 0)
SampleX = (SampleX - Xmean)/Xdev

#Add X[0] == 1
SampleX = np.hstack((np.ones([100,1]),SampleX))
#PLA
W = np.random.normal(size=[3,1])

#train
for _ in range(100):
    Prediction = np.dot(SampleX,W)
    PredictionBool = np.ones_like(SampleY)
    PredictionBool[np.where(Prediction < 0)[0]] = -1
    #record fault rate
    number = len(np.where(PredictionBool != SampleY)[0])
    print ("Fault number: {0}".format(number))
    if number == 0:
        break
    else:
        #update W
        fixPoint = np.where(PredictionBool != SampleY)[0][0]
        W += SampleY[fixPoint] * SampleX[fixPoint,:].reshape([3,1])



#plot
fig = plt.figure()
a = fig.add_subplot(1,1,1)
a.scatter(SampleX[:50,1],SampleX[:50,2],color='blue',marker='o',label = 'Positive')
a.scatter(SampleX[50:,1],SampleX[50:,2],color='red',marker='x',label = 'Negative')
a.set_xlabel('Feature1')
a.set_ylabel('Feature2')
a.legend(loc = 'upper left')
a.set_title('Original Data')

#what the hell??????
X1 = -2
Y1 = -1/W[2] * (W[0]*1 + W[1]*X1)
X2 = 2
Y2 = -1/W[2] * (W[0]*1 + W[1]*X2)
a.plot([X1,Y1],[X2,Y2])

plt.show()







