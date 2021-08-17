#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import os


def check_balance(y_train, y_test):
    
    c, n =np.unique(y_train, return_counts=True)
    for cl, nu in zip(c,n):
        print('Class {} have {} instances in y_train.'.format(cl,nu))
    
    print()
    c, n =np.unique(y_test, return_counts=True)
    for cl, nu in zip(c,n):
        print('Class {} have {} instances in y_test.'.format(cl,nu))
        
def visulaize(x_train):
    fig, ax = plt.subplots(3,3)
    ax = ax.flatten()
    for i,j in enumerate(ax): 
        j.imshow(x_train[i])
        
        
def preprocess(x_train,y_train, x_test, y_test):
    
    x_train = x_train.reshape(-1, 28*28)
    x_test = x_test.reshape(-1, 28*28)
    
    #Scaling the image data
    stdscaler = StandardScaler()
    stdscaler_fit = stdscaler.fit(x_train)
    x_train_scaled = stdscaler_fit.transform(x_train)
    x_test_scaled = stdscaler_fit.transform(x_test)
    x_train_scaled = x_train_scaled.astype(dtype = 'float32', copy=False)
    x_test_scaled = x_test_scaled.astype(dtype = 'float32', copy=False)
    
    # One-hot encoding of y
    y_onehot_train = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
    y_onehot_test = tf.keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')
    
    return x_train_scaled, y_onehot_train, x_test_scaled, y_onehot_test

def creating_val_data(x_train_scaled, y_onehot_train, number_of_sampels = 5000, random_seed=100):

    x_train_scaled, y_onehot_train = shuffle(x_train_scaled, y_onehot_train, random_state = random_seed)
    x_valid_scaled = x_train_scaled[:number_of_sampels]
    y_onehot_valid =  y_onehot_train[:number_of_sampels]
    x_train_scaled = x_train_scaled[number_of_sampels:]
    y_onehot_train =  y_onehot_train[number_of_sampels:]
    
    return x_train_scaled, y_onehot_train, x_valid_scaled, y_onehot_valid

def creating_train_val_test_datasets(x_train_scaled, y_onehot_train,
                                     x_test_scaled, y_onehot_test,
                                     x_valid_scaled, y_onehot_valid, 
                                     minibatch = 60, buffersize= 60000, random_seed=100):
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_scaled, y_onehot_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_scaled, y_onehot_test))
    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid_scaled, y_onehot_valid))
    
    
    train_dataset = train_dataset.shuffle(buffer_size=buffersize, seed=random_seed, reshuffle_each_iteration=True)
    train_dataset = train_dataset.batch(minibatch, drop_remainder = True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    
    valid_dataset = valid_dataset.shuffle(buffer_size=buffersize, seed=random_seed, reshuffle_each_iteration=False)
    valid_dataset = valid_dataset.batch(batch_size=minibatch).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    
    test_dataset = test_dataset.shuffle(buffer_size=buffersize, seed=random_seed,reshuffle_each_iteration=False)
    test_dataset =test_dataset.batch(batch_size=minibatch, 
                                     drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return train_dataset, valid_dataset, test_dataset


def creat_datasets(x_train, y_train, x_test, y_test, number_of_sampels = 5000,
                   random_seed=100, minibatch = 60,  buffersize= 60000):
    
    x_train_scaled, y_onehot_train, x_test_scaled, y_onehot_test = preprocess(x_train, y_train, x_test, y_test)

    # Creating validation data
    x_train_scaled, y_onehot_train, x_valid_scaled, y_onehot_valid = creating_val_data(x_train_scaled, y_onehot_train,
                                                                                       number_of_sampels = number_of_sampels,
                                                                                       random_seed=random_seed)
    # Creating Datasets
    train_dataset, valid_dataset, test_dataset = creating_train_val_test_datasets(x_train_scaled, y_onehot_train,
                                                                                  x_test_scaled, y_onehot_test,
                                                                                  x_valid_scaled, y_onehot_valid, 
                                                                                  minibatch = minibatch,
                                                                                  buffersize= buffersize, 
                                                                                  random_seed=random_seed)
    
    return train_dataset, valid_dataset, test_dataset
