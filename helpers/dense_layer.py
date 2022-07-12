#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K



class dense_layer(tf.keras.layers.Layer):
    '''
    Creating Custom Dense Layer
    '''
    def __init__(self, units, random_seed, **kwargs):
        
        super(dense_layer, self).__init__(**kwargs)
        self.units =  units
        self.random_seed =  random_seed
        
    def build(self, input_shape):
        self.kernel_weights = self.add_weight(name = 'weights', shape = (input_shape[-1], self.units),
                                              dtype = self.dtype, 
                                              initializer=tf.keras.initializers.Orthogonal(gain=1, seed=self.random_seed),
                                              trainable = True)
        
        self.bias = self.add_weight(name = 'bias', shape = (self.units,), dtype = self.dtype,
                                   initializer=tf.keras.initializers.zeros(), trainable = True)
        
    def call(self, inputs):
        
        return tf.add(tf.matmul(inputs, self.kernel_weights), self.bias)
    
    def get_config(self):
        
        config = super(dense_layer, self).get_config()
        config.update({'units': self.units})
        config.update({'random_seed': self.random_seed})
        
        return config