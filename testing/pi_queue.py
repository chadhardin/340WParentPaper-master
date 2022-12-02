#!/usr/bin/python3

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
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
import tensorflow as tf
import pandas as pd
import sys
import math
import numpy as np
import sklearn.metrics as sklm
import all_models
from sklearn import preprocessing
import os

window_size = sys.argv[1]
first_layer_num_neurons = int(sys.argv[2])
train_X = np.empty((0,157918), float)
train_y = np.empty((0,1), int)

train_X=np.load("./TRAINING_X_"+window_size+".npy")
train_y=np.load("./TRAINING_Y_"+window_size+".npy")
test_X=np.load("./TESTING_X_"+window_size+".npy")
test_y=np.load("./TESTING_Y_"+window_size+".npy")
train_y = np.asarray(train_y).astype('float32').reshape((-1,1))
window_index=0
num_windows=np.size(train_X,0)
window_size=np.size(train_X,1)
num_features=np.size(train_X,2)
test_X = test_X[:-21,:,:]
for i in range(train_X.shape[-1]):
    train_X[:,:,i] = np.interp(train_X[:,:,i], (train_X[:,:,i].min(), train_X[:,:,0:].max()), (-1, +1))
print(train_X.shape, train_y.shape, test_X.shape, train_y.shape)

model = all_models.get_optfastrnnlstm_single_layer([None, window_size, num_features], dropout = 0.6, first_layer_neurons=first_layer_num_neurons)

model.fit(x = train_X,y = train_y, epochs = 10, batch_size = 100)

model.save('rnnlstmadamopt.h5')

model1 = all_models.get_optfastrnnlstm_single_layer1([None, window_size, num_features], dropout = 0.6, first_layer_neurons=first_layer_num_neurons)

model1.fit(x = train_X,y = train_y, epochs = 10, batch_size = 100)
model.save('rnnlstmdsgopt.h5')
print("Please make necessary code changes as per the dataset")

while window_index < num_windows :
    test_x = test_X[window_index:window_index+1,:,:]
    y=model.predict(test_x)
    window_index=window_index+1
    print("SAMPLE #:", window_index)

while window_index < num_windows :
    test_x = test_X[window_index:window_index+1,:,:]
    y1=model1.predict(test_x)
    window_index=window_index+1
    print("SAMPLE #:", window_index)

print(y)
print(y1)
