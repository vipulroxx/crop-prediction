#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
import os, csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import math
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
print(' /--------------------- Loading Training Data ---------------------/ ')
data = pd.read_csv('odisha.csv')
df = pd.DataFrame(data = data)
print(df)
df = df.loc[ :, 'Rainfall':'PET'].values.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
values = min_max_scaler.fit_transform(df)
df_normalized = pd.DataFrame(values)
print(' /--------------------- Normalizing Training Data ---------------------/ ')
print(df_normalized)
y = df_normalized[0]
X = df_normalized[[1,2,3,4,5,6]]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


print('\nX_train\n', X_train,'\nX_test\n',X_test,'\nY_train\n ',Y_train,'\nY_test\n ',Y_test)
