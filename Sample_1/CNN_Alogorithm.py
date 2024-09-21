# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 22:16:44 2024

@author: MUDDANA LAKSHMI
"""

#IMPORTING NECESSARY LIBRARIES
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
import matplotlib.pyplot as plt
import random

#IMPORTING THE DATASETS
x_train=np.loadtxt('D:\Machine_Learning_Samples\Sample_1\Image Classification CNN Keras Dataset-20240327T085101Z-001\Image Classification CNN Keras Dataset\input.csv',delimiter=',')
y_train=np.loadtxt('D:\Machine_Learning_Samples\Sample_1\Image Classification CNN Keras Dataset-20240327T085101Z-001\Image Classification CNN Keras Dataset\labels.csv',delimiter=',')
x_test=np.loadtxt('D:\Machine_Learning_Samples\Sample_1\Image Classification CNN Keras Dataset-20240327T085101Z-001\Image Classification CNN Keras Dataset\input_test.csv',delimiter=',')
y_test=np.loadtxt('D:\Machine_Learning_Samples\Sample_1\Image Classification CNN Keras Dataset-20240327T085101Z-001\Image Classification CNN Keras Dataset\labels_test.csv',delimiter=',')


#CHECKING THE SHAPE OF THE DATASETS
print("Shape of x_train",x_train.shape)
print("shape of y_train",y_train.shape)
print("shape of x_test",x_test.shape)
print("shape of y_test",y_test.shape)

#RESHAPING THE DATASETS TO 2000(TRAINING_SET) AND 400(TESTING SET)
x_train=x_train.reshape(len(x_train),100,100,3)
y_train=y_train.reshape(len(y_train),1)
x_test=x_test.reshape(len(x_test),100,100,3)
y_test=y_test.reshape(len(y_test),1)

x_train[1,:] 

x_train=x_train/255.0
x_test=x_test/255.0

#PLOTING THE RANDOM IMAGE IN THE DATASET
'''idx=x_train.any()
plt.show(x_train[idx,:])
plt.show()'''

#MODEL BUILDING
model=Sequential([Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)),MaxPooling2D((2,2)),
Conv2D(32,(3,3),activation='relu'),
MaxPooling2D((2,2)),

Flatten(),
Dense(64,activation='relu'),
Dense(1,activation='sigmoid')])

