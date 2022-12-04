#!/usr/bin/python3
''' Import all of the necessary '''
from math import sqrt
from numpy import concatenate
import numpy as np
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import load_model
from keras.utils import CustomObjectScope
import tensorflow as tf
import pandas as pd
import sys
import math
import numpy as np
import all_models
import os

'''Get inputs from user window_size = 100, first_layer_num_neurons = 128'''
window_size = str(sys.argv[1])
first_layer_num_neurons = int(sys.argv[2])

'''Training data loading and pre-processing (if model hasn't been trained and don't have a saved model run these)'''
train_X=np.load("./TRAINING_X_"+window_size+".npy")
train_y=np.load("./TRAINING_Y_"+window_size+".npy")
train_y = np.asarray(train_y).astype('float32').reshape((-1,1))
num_windows=np.size(train_X,0)
window_size=np.size(train_X,1)
num_features=np.size(train_X,2)
for i in range(train_X.shape[-1]):
    train_X[:,:,i] = np.interp(train_X[:,:,i], (train_X[:,:,i].min(), train_X[:,:,0:].max()), (-1, +1))

'''Load testing data'''
test_X=np.load("./TESTING_X_{}.npy".format(window_size))
test_y=np.load("./TESTING_Y_{}.npy".format(window_size))
test_y = np.asarray(test_y).astype('float32').reshape((-1,1))
for i in range(test_X.shape[-1]):
    test_X[:,:,i] = np.interp(test_X[:,:,i], (test_X[:,:,i].min(), test_X[:,:,0:].max()), (-1, +1))
num_windows=np.size(test_X,0)
window_size=np.size(test_X,1)
num_features=np.size(test_X,2)
'''Make sure the shape is correct'''
print(train_X.shape, train_y.shape, test_X.shape, train_y.shape)


'''Now it is time to create a checkpoint path for each epoch of the training'''
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

'''This model uses Adam Optimizer, which while testing was not as good as the SGD (Standard Gradient Descent)'''
#model = all_models.get_optfastrnnlstm_single_layer([None, window_size, num_features], dropout = 0.6, first_layer_neurons=first_layer_num_neurons)
#model.fit(x = train_X,y = train_y, epochs = 10, batch_size = 100, callbacks=[cp_callback])

'''This model uses SGD for optimization, so we need to select the model then load the weights in from the saved epochs'''
model = all_models.get_optfastrnnlstm_single_layer1([None, window_size, num_features], dropout = 0.1, first_layer_neurons=first_layer_num_neurons)
model.fit(x = train_X,y = train_y, epochs = 9, batch_size = 100, callbacks=[cp_callback], validation_split=0.1)
#latest = tf.train.latest_checkpoint(checkpoint_dir)

'''Testing of the model is done here'''
loss, acc = model.evaluate(test_X, test_y, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
#print(model.metrics_names)

num_windows = np.size(test_X, 0)
#model.load_weights(latest)
y=model.predict(test_X, batch_size = 100)
y = y[:,0,:]
thresh = .8
for i in range(y.shape[1]):
    y[y[:,i] >= thresh] = 1
    y[y[:,i] < thresh] = 0

from sklearn.metrics import precision_score, accuracy_score, recall_score, cohen_kappa_score, f1_score
acc, prec, re, kappa, f1 = precision_score(test_y, y), accuracy_score(test_y, y), recall_score(test_y, y), cohen_kappa_score(test_y, y), f1_score(test_y, y)

print("Accuracy is {}, Precision is {}, Recall is {}, Kappa_score is {}, and f1_score is {}".format(acc, prec, re, kappa, f1))



