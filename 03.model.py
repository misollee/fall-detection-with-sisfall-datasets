# importing the required libraries ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import pandas as pd
import numpy as np
import random

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense


# setup ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
os.chdir('/home/mslee/fall_detection')
pd.set_option('display.max_rows', None)

# load data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
total_df = pd.read_csv('data/total_df.csv')
total_df['key'] = total_df.id + total_df.activity 

# select variables ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
features = total_df.dtypes
features =  features.index[features != 'object']
features = [x.replace('.1','') for x in features]
features = list(set(features))
features = np.setdiff1d(features, ['fall','window'])

# split train / test ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

split_key = list(total_df.key.unique())
samp_tr = random.sample(split_key, round(len(split_key)*0.8))
samp_ts = np.setdiff1d(split_key, samp_tr)

tr_df = total_df[total_df.key.isin(samp_tr)]
ts_df = total_df[total_df.key.isin(samp_ts)]

x_tr = tr_df[features].to_numpy()
x_ts = ts_df[features].to_numpy()

y_tr = tr_df['activity']
y_ts = ts_df['activity']

# scaling ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

scaler_ = MinMaxScaler()
scaler_.fit(x_tr)
x_tr_scaled_ = scaler_.transform(x_tr)
x_ts_scaled_ = scaler_.transform(x_ts)

# reshaping data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
x_train = x_tr.reshape((x_tr.shape[0], 1, x_tr.shape[1], 1))
x_test = x_ts.reshape((x_ts.shape[0], 1, x_ts.shape[1], 1))

#checking the shape after reshaping
print(x_train.shape)
print(x_test.shape)

# 모델 함수화해서 분리하기 
#defining model
model=Sequential()
#adding convolution layer
model.add(Conv2D(5,(3,3),activation='relu', input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])))
#adding pooling layer
model.add(MaxPool2D(2,2))
#adding fully connected layer
model.add(Flatten())
model.add(Dense(125, activation='relu'))
#adding output layer
model.add(Dense(y_tr.nunique(), activation='softmax'))
#compiling the model
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#fitting the model
model.fit(x_train, y_tr, epochs=100)