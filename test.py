import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv1D
from keras.layers.normalization.batch_normalization import BatchNormalization
from tensorflow.python.keras.layers import  CuDNNGRU,  CuDNNLSTM
from tensorflow.python.keras.layers import LSTM,GRU

import pandas as pd

def all_models(shape,dropout):
    def get_cudnngru(shape,dropout):
        model = Sequential()
        with tf.variable_scope("CuDNNGRU1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNGRU(64,input_shape=(shape),return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("CuDNNGRU2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNGRU(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("CuDNNGRU3" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNGRU(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("CuDNNGRU4" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNGRU(64,return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))
        
        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
            
        return model
       
        
    def get_cudnnlstm(shape,dropout):
        model = Sequential()
        with tf.variable_scope("CuDNNLSTM1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,input_shape=(shape),return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("CuDNNLSTM2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("CuDNNLSTM3" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("CuDNNLSTM4" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
        return model


    def get_cudnn3lstm(shape,dropout):
        model = Sequential()
        with tf.variable_scope("CuDNNLSTM1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,input_shape=(shape),return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("CuDNNLSTM2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("CuDNNLSTM3" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("CuDNNLSTM4" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("CuDNNLSTM5" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("CuDNNLSTM6" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
        return model
        
        
        

    def get_cudnncnnlstm(shape,dropout):
        model = Sequential()
        with tf.variable_scope("Conv1D1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Conv1D(128, input_shape=(train_X.shape[1:]), kernel_size=3, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
        with tf.variable_scope("Conv1D2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Conv1D(128,kernel_size=3, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

        with tf.variable_scope("CuDNNLSTM1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("CuDNNLSTM2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("CuDNNLSTM3" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("CuDNNLSTM4" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(CuDNNLSTM(64,return_sequences=False))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
        return model
        
 
    def get_fastrnnlstm(shape,dropout):
        model = Sequential()
        with tf.variable_scope("LSTM1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(64,input_shape=(shape),return_sequences=True, celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("LSTM2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(64,return_sequences=True,celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("LSTM3" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(64,return_sequences=True,celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("LSTM4" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(64,return_sequences=False,celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
        return model
        
        
    def get_fastgrnnlstm(shape,dropout):
        model = Sequential()
        with tf.variable_scope("LSTM1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(64,input_shape=(shape),return_sequences=True, celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("LSTM2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(64,return_sequences=True,celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("LSTM3" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(64,return_sequences=True,celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("LSTM4" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(64,return_sequences=False,celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
        return model
        
        
    def get_optfastgrnnlstm(shape,dropout):
        model = Sequential()
        with tf.variable_scope("LSTM1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(128,input_shape=(shape),return_sequences=True, celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("LSTM2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(64,return_sequences=False,celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(32, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()

        return model
        
    def get_optfastrnnlstm(shape,dropout):
        model = Sequential()
        with tf.variable_scope("LSTM1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(128,input_shape=(shape),return_sequences=True, celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("LSTM2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(LSTM(64,return_sequences=False,celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(32, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
        return model


 
    def get_fastrnngru(shape,dropout):
        model = Sequential()
        with tf.variable_scope("GRU1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(64,input_shape=(shape),return_sequences=True, celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("GRU2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(64,return_sequences=True,celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("GRU3" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(64,return_sequences=True,celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("GRU4" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(64,return_sequences=False,celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
        return model
        
        
    def get_fastgrnngru(shape,dropout):
        model = Sequential()
        with tf.variable_scope("GRU1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(64,input_shape=(shape),return_sequences=True, celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("GRU2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(64,return_sequences=True,celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("GRU3" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(64,return_sequences=True,celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

        with tf.variable_scope("GRU4" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(64,return_sequences=False,celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
        return model
        
        
    def get_optfastgrnngru(shape,dropout):
        model = Sequential()
        with tf.variable_scope("GRU1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(128,input_shape=(shape),return_sequences=True, celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("GRU2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(64,return_sequences=False,celltype="FastGRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(32, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()

        return model
        
    def get_optfastrnngru(shape,dropout):
        model = Sequential()
        with tf.variable_scope("GRU1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(128,input_shape=(shape),return_sequences=True, celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	
            
        with tf.variable_scope("GRU2" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(GRU(64,return_sequences=False,celltype="FastRNNCell"))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))	

            
        with tf.variable_scope("DENSE1" ,reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(32, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        with tf.variable_scope("DENSE2", reuse=tf.AUTO_REUSE) as scope:
            model.add(Dense(1, activation='sigmoid'))
            opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-3)
            model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
            model.summary()
        return model

from math import sqrt
from numpy import concatenate
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv1D
from keras.layers.normalization.batch_normalization import BatchNormalization
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf
import pandas as pd
import time
import random
import sys
import recurrent_fast

 
 
import keras
import numpy as np
import sklearn.metrics as sklm

print("Please make necessary code changes as per the dataset")
# change shape if selected feature dataset is used
test_X=np.empty((0,53),float)
test_y=np.empty((0,1),int)

model=all_models.GRU([100,53],0.7) #(shape,dropout) in accordance to dataset

i=0
with open(sys.argv[1]) as f:
    lines=f.readlines()
    for line in lines:
        myarray = np.fromstring(line, dtype=float, sep=',')
        if myarray.size!=0:
            test_y=np.array([myarray[-1]])
            myarray=myarray[:-1]
            test_X=np.append(test_X,[myarray],axis=0)
            i+=1
            if(i==100):
                y=model.predict(np.reshape(test_X,[1,100,53]))
                print(y,test_y)
                test_X=np.delete(test_X,0,axis=0)
                test_y=np.empty((0,1),int)
                i=99
                