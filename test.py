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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow as tf
import pandas as pd
import time
import random
import sys
import recurrent_fast
from training import all_models
import keras
import numpy as np
import sklearn.metrics as sklm

data = pd.read_csv("training.csv", header = None)
print(data.shape)

print("Please make necessary code changes as per the dataset")
# change shape if selected feature dataset is used
test_X=np.empty((0,79),float)
test_y=np.empty((0,1),int)

model=all_models.get_fastrnnlstm([100,79],0.7) #(shape,dropout) in accordance to dataset

i=0
d=0
with open(sys.argv[1]) as f:
    lines=f.readlines()
    for line in lines:
        if d < 800:
            myarray = np.fromstring(line, sep=',')
            if myarray.size!=0:
                test_y=np.array([myarray[-1]])
                myarray=myarray[:-1]
                test_X=np.append(test_X,[myarray],axis=0)
                i+=1
                if(i==100):
                    y=model.predict(np.reshape(test_X,[1,100,79]))
                    print(y,test_y)
                    test_X=np.delete(test_X,0,axis=0)
                    test_y=np.empty((0,1),int)
                    i=99
        d+= 1
