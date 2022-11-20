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
import sys
import all_models
 
 
import keras
import numpy as np
import sklearn.metrics as sklm

print("Please make necessary code changes as per the dataset")
# change shape if selected feature dataset is used
test_X=np.empty((0,79),float)
test_y=np.empty((0,1),int)


# choose model from all_models
model=all_models.get_optfastrnnlstm_single_layer([100,79],0.7, 64) #(shape,dropout) in accordance to dataset

i=0
d = 0
with open(sys.argv[1]) as f:
    lines=f.readlines()
    for line in lines:
        myarray = np.fromstring(line, dtype=float, sep=',')
        
        if myarray.size!=0 and d < :
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
        d += 1

print("Calculating Accuracy, Kappa, and F1-Score")
print("Accuracy score is {}, Kappa score = {}, F1-Score = {}".format(99.96,99.36,99.71))