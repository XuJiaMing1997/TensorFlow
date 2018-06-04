import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Hyperparameter
lr = 0.01
time_steps = 30#bigger value may give more imformation and more easier get convergence
batch_num = 20
row_data_start = 0
single_input_size = 1
cell_size = 20
output_size = 1

#Synthesize data
def getData():
    global time_steps,batch_num,row_data_start
    getX = np.arange(row_data_start,row_data_start + batch_num*time_steps).reshape([batch_num,time_steps])/\
           (np.pi*10)
    #getX = [batch_num,time_steps]
    targetY = np.sin(getX)
    inputY = np.cos(getX)
    '''
    f = plt.figure()
    a = f.add_subplot(1,1,1)
    a.plot(getX,targetY,'r',getX,inputY,'b--')
    plt.show()
    '''
    row_data_start += time_steps
    #getX = [batch_num,time_steps],inputY = targetY = [batch_num,time_steps,1]
    return getX,inputY[:,:,np.newaxis],targetY[:,:,np.newaxis]

class LSTM_RNN(object):
    def __init__(self,batch_num,time_steps,single_input_size,output_size,cell_size,lr):
        #Variables Initialization
        self.batch_num = batch_num
        self.time_steps = time_steps
        self.single_input_size = single_input_size
        self.cell_size = cell_size
        self.output_size = output_size
        self.lr = lr
        self.Weights = {
            'inputLayer':tf.Variable(tf.random_normal([single_input_size,cell_size],stddev=0.5)),
            'outputLayer':tf.Variable(tf.random_normal([cell_size,output_size],stddev=0.5))
        }
        self.biases = {
            'inputLayer':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[cell_size])),
            'outputLayer':tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[output_size]))
        }
        #input Interface
        self.X = tf.placeholder(tf.float32,[None,time_steps,single_input_size],name='RNN_InputX')
        self.Y = tf.placeholder(tf.float32,[None,time_steps,output_size],name='RNN_InputY')
        #structure Initialization
        self.addInputLayer()
        self.addCellLayer()
        self.addOutputLayer()
        self.computeCost()
        self.train()

    def addInputLayer(self):
        #self.X = [batch_num * time_steps,1]
        Combination = tf.matmul(tf.reshape(self.X,[-1,self.single_input_size]),self.Weights['inputLayer'])\
                      + self.biases['inputLayer']
        #inputLayer output = [batch_num,time_steps,cell_size]
        self.inputLayerRes = tf.reshape(Combination,[-1,self.time_steps,self.cell_size])

    def addCellLayer(self):
        #initialization cell
        initCell =  tf.nn.rnn_cell.BasicLSTMCell(self.cell_size,forget_bias=1.0,state_is_tuple=True)
        self.initState = initCell.zero_state(self.batch_num,dtype=tf.float32)
        #cellOutput = [batch_num,time_steps,cell_size]
        self.cellOutput,self.finalState = tf.nn.dynamic_rnn(initCell,self.inputLayerRes,
                                                            initial_state=self.initState,time_major=False)

    def addOutputLayer(self):
        #cellOutput = [batch_num*time_steps,cell_size]
        self.Prediction = tf.matmul(tf.reshape(self.cellOutput,[-1,self.cell_size]),self.Weights['outputLayer'])\
                      +self.biases['outputLayer']

    #what the hell??????-----copy from Morvan
    def computeCost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.Prediction, [-1], name='reshape_pred')],
            [tf.reshape(self.Y, [-1], name='reshape_target')],
            [tf.ones([self.batch_num * self.time_steps], dtype=tf.float32)],
                average_across_timesteps=True,
                softmax_loss_function=self.ms_error,
            name='losses'
        )
        self.loss = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_num
        )

    def train(self):
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    #What the hell???????
    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

if __name__ == '__main__':
    model = LSTM_RNN(batch_num,time_steps,single_input_size,output_size,cell_size,lr)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(100):
            X,inputY,targetY = getData()
            if i == 0:
                feed_dict = {
                    model.X:inputY,
                    model.Y:targetY
                }
            else:
                feed_dict = {
                    model.X:inputY,
                    model.Y:targetY,
                    model.initState:lastState
                }
            _,loss,pred,lastState = sess.run([model.train_step,model.loss,model.Prediction,model.finalState],
                                             feed_dict=feed_dict)

            #plot---copy from Morvan
            plt.ion()
            plt.show()
            line= plt.plot(X[0,:],targetY[0].flatten(),'r',X[0,:],pred[:time_steps],'b--')
            plt.ylim((-1.2, 1.2))
            plt.draw()
            plt.pause(0.3)

            if i % 20 == 0:
                print("Cost:",loss)
                #use plt.clf clean very former figure-----XCP use this
                plt.clf()
        plt.ioff()
        plt.pause(100)




