#!/usr/bin/env python
# coding: utf-8

from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from tensorflow.keras import backend as K


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle






# Callback for saving best model
def save_best_model_callback(model_name):
    path = './models/'+ model_name
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)
        
    savemodel_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(path,'best_model.h5'),
                                                      mode = 'min', monitor = 'val_loss', save_best_only=True)
    return savemodel_cb

######################################################################################################################

# Callback for Visualizing in TensorBoard
def tensorboard_callback(model_name):
    path = './logs/'+ model_name
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)
        
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir = path, 
                                                    histogram_freq=0,
                                                    write_graph=True, write_images=True,
                                                   profile_batch = 100000000)
    return tensorboard_cb



######################################################################################################################

def check_balance(y_train, y_test):
    
    c, n = np.unique(y_train, return_counts=True)
    for cl, nu in zip(c,n):
        print('Class {} have {} instances in y_train.'.format(cl,nu))
    
    print()
    c, n = np.unique(y_test, return_counts=True)
    for cl, nu in zip(c,n):
        print('Class {} have {} instances in y_test.'.format(cl,nu))
######################################################################################################################
        
def visualize(x_train):
    fig, ax = plt.subplots(3,3)
    ax = ax.flatten()
    for i,j in enumerate(ax): 
        j.imshow(x_train[i])
        
######################################################################################################################
        
def preprocess(x_train,y_train, x_test, y_test, num_classes, reshape_to ):
    
    x_train = x_train.reshape(-1, reshape_to )
    x_test = x_test.reshape(-1, reshape_to)
    
    #Scaling the image data
    stdscaler = StandardScaler()
    stdscaler_fit = stdscaler.fit(x_train)
    x_train_scaled = stdscaler_fit.transform(x_train)
    x_test_scaled = stdscaler_fit.transform(x_test)
    x_train_scaled = x_train_scaled.astype(dtype = 'float32', copy=False)
    x_test_scaled = x_test_scaled.astype(dtype = 'float32', copy=False)

    # One-hot encoding of y
    y_onehot_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes, dtype='float32')
    y_onehot_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes, dtype='float32')
    
    return x_train_scaled, y_onehot_train, x_test_scaled, y_onehot_test
######################################################################################################################


def creating_val_data(x_train_scaled, y_onehot_train, number_valid_sampels = 5000, random_seed=100):

    x_train_scaled, y_onehot_train = shuffle(x_train_scaled, y_onehot_train, random_state = random_seed)
    x_valid_scaled = x_train_scaled[:number_valid_sampels]
    y_onehot_valid =  y_onehot_train[:number_valid_sampels]
    x_train_scaled = x_train_scaled[number_valid_sampels:]
    y_onehot_train =  y_onehot_train[number_valid_sampels:]
    
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

######################################################################################################################

def creat_datasets(x_train, y_train, x_test, y_test, number_valid_sampels = 5000,
                   random_seed=100, minibatch = 60,  buffersize= 60000, num_classes=10,
                   reshape_to = 28*28 , back_reshape = (28,28)):
    

    x_train_scaled, y_onehot_train, x_test_scaled, y_onehot_test = preprocess(x_train, y_train, x_test, y_test,
                                                                               num_classes, reshape_to)

    # Creating validation data
    x_train_scaled, y_onehot_train, x_valid_scaled, y_onehot_valid = creating_val_data(x_train_scaled, y_onehot_train,
                                                                                       number_valid_sampels = number_valid_sampels,
                                                                                       random_seed=random_seed)
    if back_reshape!= None:
        x_train_scaled = x_train_scaled.reshape(back_reshape)
        x_valid_scaled = x_valid_scaled.reshape(back_reshape)
        x_test_scaled = x_test_scaled.reshape(back_reshape)

        
    # Creating Datasets
    train_dataset, valid_dataset, test_dataset = creating_train_val_test_datasets(x_train_scaled, y_onehot_train,
                                                                                  x_test_scaled, y_onehot_test,
                                                                                  x_valid_scaled, y_onehot_valid, 
                                                                                  minibatch = minibatch,
                                                                                  buffersize= buffersize, 
                                                                                  random_seed=random_seed)
    
    return train_dataset, valid_dataset, test_dataset

######################################################################################################################


def reset_graph():
    tf.keras.backend.clear_session()
    print('session is clear')

######################################################################################################################
   
def read_pickle_file(filename = "sorted_evaluation.pkl"):
    file = open(filename, "rb")
    output = pickle.load(file)
    return output

######################################################################################################################

def write_pickle_file(filename, content):
    with open(filename, 'wb') as f:
        pickle.dump(content, f)
######################################################################################################################

def grid_serach(model_func, test_dataset, 
                batch_size, sort=True,
                save_eval_path = "sorted_evaluation.pkl",
                weights_path ="pretrained_weights.h5",
                loss =  tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics = tf.keras.metrics.CategoricalAccuracy(),
               ):
    
    
    b_mm = [True,False]
    b_mv = [True,False]
    f_mm = [True,False]
    f_mv = [True,False]

    evaluation = {}
    for bmm in b_mm:
        for bmv in b_mv:
            for fmm in f_mm:
                for fmv in f_mv:
                    model_bln_layer = model_func(b_mm = bmm, b_mv=bmv, f_mm = fmm, f_mv=fmv, batch_size = batch_size)

                    model_bln_layer.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
                                                   loss = loss,
                                                   metrics = [metrics])
                    
                    model_bln_layer.load_weights(weights_path)

                    name = 'Bmm_' + str(bmm) + ' Bmv_' + str(bmv) + ' Fmm_' + str(fmm) + ' Fmv_' + str(fmv)
                    evaluation[ name ] = model_bln_layer.evaluate(test_dataset)
                    print(evaluation)

                    reset_graph()
                    del model_bln_layer, name
                                   

    if sort:
        evaluation = sorted(evaluation.items(), key=lambda x:(x[1][0], x[1][1]))
        
    if save_eval_path != None:
        file = open(save_eval_path, "wb")
        pickle.dump(evaluation, file)
        file.close()
         
    return  evaluation         
                





