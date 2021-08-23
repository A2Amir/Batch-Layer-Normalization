#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


def reset_random_seeds(random_seed = 100):
    os.environ['PYTHONHASHSEED']=str(random_seed)
    tf.random.set_seed(random_seed)


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
    
    
    train_dataset = train_dataset.shuffle(buffer_size=buffersize, seed=random_seed, reshuffle_each_iteration=False)
    train_dataset = train_dataset.batch(minibatch,
                                        drop_remainder = True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    
    valid_dataset = valid_dataset.shuffle(buffer_size=buffersize, seed=random_seed, reshuffle_each_iteration=False)
    valid_dataset = valid_dataset.batch(batch_size=minibatch,
                                         drop_remainder=True).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    
    test_dataset = test_dataset.shuffle(buffer_size=buffersize, seed=random_seed, reshuffle_each_iteration=False)
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

def reset_graph():
    tf.keras.backend.clear_session()
    print('session is clear')

    
def read_pick_file(filename = "sorted_evaluation.pkl"):
    file = open(filename, "rb")
    output = pickle.load(file)
    return output

def grid_serach(model_func, callback, train_dataset, valid_dataset, test_dataset,
                epochs, sort=True, filename = "sorted_evaluation.pkl", ):
    
        
    b_mm = [True,False]
    b_mv = [True,False]
    f_mm = [True,False]
    f_mv = [True,False]
    
    evaluation = {}
    for bmm in b_mm:
        for bmv in b_mv:
            for fmm in f_mm:
                for fmv in f_mv:
                    
                    model_custom_bln_layer = model_func(inputshape = (784,), units1 = 100, units2 =100, units3=100,
                         classes=10, random_seed=100, batch_size= 60, b_mm = bmm, b_mv=bmv, f_mm = fmm, f_mv=fmv)

                    # Callback for resetting moving mean and variances at the end of each epoch
                    custom_bln_layer_cb = callback()

                    model_custom_bln_layer.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
                                   loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                                   metrics = [tf.keras.metrics.CategoricalAccuracy()])

                    model_custom_bln_layer_history =  model_custom_bln_layer.fit(train_dataset.take(400),
                                                                                 epochs=epochs, verbose=1,
                                                                                 callbacks=[custom_bln_layer_cb],
                                                                                 validation_data=valid_dataset.take(50),
                                                                                 shuffle=True)
                    
                    name = 'Bmm_' + str(bmm) + ' Bmv_' + str(bmv) + ' Fmm_' + str(fmm) + ' Fmv_' + str(fmv)
                    evaluation[ name ] = model_custom_bln_layer.evaluate(test_dataset)
                    print(evaluation)
                    
                    reset_graph()
                    del model_custom_bln_layer,  custom_bln_layer_cb, model_custom_bln_layer_history, name
                    
    #reset_graph()               

    if sort:
        evaluation = sorted(evaluation.items(), key=lambda x:(x[1][0], x[1][1]))
        
    if filename != None:
        file = open(filename, "wb")
        pickle.dump(evaluation, file)
        file.close()
         
    return  evaluation         
                



