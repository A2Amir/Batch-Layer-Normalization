#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


 ########################################################################################################

class bln_layer(tf.keras.layers.Layer):   
    """
    developing the new normalization method termed Batch Layer normalization, a novel normalization method to speed the convergence for various neural network models. Unlike batch and layer normalization methods that use the distribution of summed inputs to a neuron for calculating a mean and variance over a mini-batch and directly in a layer, respectively, the proposed method estimates the normalization statistics from the summed inputs to the neurons within a hidden layer via a mini-batch and directly in a layer together.

    """
    
    def __init__(self, stateful, batchsize,
                 batch_moving_mean=True, batch_moving_var=True,
                 feature_moving_mean=False, feature_moving_var=False,
                 **kwargs):
        
        super(bln_layer, self).__init__(**kwargs)
        self.stateful = stateful
        self.batchsize = batchsize
        self.batch_size = tf.cast(self.batchsize, 'float32')
        
        self.batch_moving_mean = batch_moving_mean
        self.batch_moving_var = batch_moving_var
        
        self.feature_moving_mean = feature_moving_mean
        self.feature_moving_var = feature_moving_var

    def build(self, input_shape):
        
        shape = input_shape[-1:]
        self.dk = tf.cast(input_shape[-1], 'float32')
        
        if len(input_shape) == 2:
            bn_shape = (1, input_shape[1])
            feature_shape = (self.batchsize,1)
           
        elif len(input_shape) == 3:
            bn_shape = (1,input_shape[1], input_shape[2])
            feature_shape = (self.batchsize, input_shape[1], 1)
            
        elif len(input_shape) == 4:
            bn_shape = (1,input_shape[1], input_shape[2], input_shape[3])
            feature_shape = (self.batchsize, input_shape[1],input_shape[2], 1)
           
        else:
            print('layer shape must be 2D or 3D or 4D')

        self.gamma1 = self.add_weight(name = 'scale1', shape = shape,
                                     initializer = tf.keras.initializers.ones(),
                                     trainable = True)
       
        self.beta1 = self.add_weight(name = 'shift1', shape = shape,
                                    initializer = tf.keras.initializers.zeros(),
                                   trainable = True)

    
        self.offset = tf.Variable(0.001, dtype = 'float32', trainable=False)

        #batch_moving_mean
        self.moving_Bmean = self.add_weight(name = 'moving_Bmean', shape = bn_shape,
                                            initializer = tf.keras.initializers.Zeros(),
                                            trainable = False)
        #batch_moving_var
        self.moving_Bvar =  self.add_weight(name = 'moving_Bvar', shape = bn_shape ,
                                            initializer = tf.keras.initializers.Zeros(),
                                            trainable = False)
        #feature_moving_mean
        self.moving_Fmean = self.add_weight(name = 'moving_Fmean', shape = feature_shape,
                                            initializer = tf.keras.initializers.Zeros(),
                                            trainable = False)
        #feature_moving_var
        self.moving_Fvar =  self.add_weight(name = 'moving_Fvar', shape = feature_shape ,
                                            initializer = tf.keras.initializers.Zeros(),
                                            trainable = False)        
        
        self.batch_count = tf.Variable(0, dtype = 'float32', name = 'batchcount', trainable=False)

        self.init_mBm = self.moving_Bmean.read_value()
        self.init_mBv = self.moving_Bvar.read_value()
        
        self.init_mFm = self.moving_Fmean.read_value()
        self.init_mFv = self.moving_Fvar.read_value()

        
    def bn_training(self, inputs):
        
        batch_mean, batch_var = tf.nn.moments(inputs, axes = [0], keepdims=True)
        batch_std = K.sqrt(batch_var + self.offset)
        self.moving_Bmean.assign_add(batch_mean)
        self.moving_Bvar.assign_add(batch_var)
        

        feature_mean, feature_var = tf.nn.moments(inputs, axes = [-1], keepdims=True)
        feature_std = K.sqrt(feature_var + self.offset )
        self.moving_Fmean.assign_add(feature_mean)
        self.moving_Fvar.assign_add(feature_var)
        
     
        x_f =   (inputs - feature_mean) / feature_std 
        x_b = (inputs - batch_mean) / batch_std
        numerator =  (x_f * ((1 / self.batch_size) - .0001)) + (x_b * (1- ((1 / self.batch_size) + .0001)))
        x_beta  =  numerator / tf.math.sqrt(self.dk) 

        output =   (self.gamma1 * (x_beta))+ self.beta1
        
        return output
    

    
    def update_mm_mv(self):
        """
        Updating batch_moving_mean and batch_moving_var, feature_moving_mean and feature_moving_var at the end of epoch
        """        
        self.moving_Bmean.assign(tf.cond(tf.greater(self.batch_count,0), 
                                        lambda: tf.divide(self.moving_Bmean,self.batch_count), lambda: self.moving_Bmean,
                                         name='update_mBm'))
        
        self.moving_Bvar.assign(tf.cond(tf.greater(self.batch_count,0), 
                                       lambda: tf.multiply(self.moving_Bvar,
                                                           tf.divide(self.batch_size,
                                                                     tf.multiply(tf.subtract(self.batch_size,1),
                                                                                 self.batch_count))),
                                       lambda: self.moving_Bvar, name='update_mBv'))
        
        self.moving_Fmean.assign(tf.cond(tf.greater(self.batch_count,0), 
                                        lambda: tf.divide(self.moving_Fmean,self.batch_count),
                                        lambda: self.moving_Fmean, name='update_mFm'))
        
        self.moving_Fvar.assign(tf.cond(tf.greater(self.batch_count,0), 
                                       lambda: tf.multiply(self.moving_Fvar,
                                                           tf.divide(self.batch_size,
                                                                     tf.multiply(tf.subtract(self.batch_size,1),
                                                                                 self.batch_count))),
                                       lambda: self.moving_Fvar, name='update_mFv'))
        
        
    def bn_inference(self, inputs):

        batch_mean , batch_std = 0, 0
        feature_mean , feature_std = 0, 0

        if (self.batch_moving_mean == False) and (self.batch_moving_var == False):
            batch_mean, batch_var = tf.nn.moments(inputs, axes =[0], keepdims=True)
            batch_std = tf.math.sqrt(tf.add(batch_var, self.offset))
            
        elif (self.batch_moving_mean == True) and (self.batch_moving_var == True):
            batch_mean = self.moving_Bmean
            batch_std = tf.math.sqrt(tf.add(self.moving_Bvar, self.offset))
            
        elif (self.batch_moving_mean == True) and (self.batch_moving_var == False):
            batch_mean = self.moving_Bmean
            _, batch_var = tf.nn.moments(inputs, axes = [0], keepdims=True)
            batch_std = tf.math.sqrt(tf.add(batch_var, self.offset))

        elif (self.batch_moving_mean == False) and (self.batch_moving_var == True):
            batch_mean, _ = tf.nn.moments(inputs, axes = [0], keepdims=True)
            batch_std = tf.math.sqrt(tf.add(self.moving_Bvar, self.offset))

            
        if (self.feature_moving_mean == False) and (self.feature_moving_var == False):
            feature_mean, feature_var = tf.nn.moments(inputs, axes = [-1], keepdims=True)
            feature_std = tf.math.sqrt(tf.add(feature_var, self.offset))
            
        elif (self.feature_moving_mean == True) and (self.feature_moving_var == True):
            feature_mean = self.moving_Fmean
            feature_std = tf.math.sqrt(tf.add(self.moving_Fvar, self.offset))

            
        elif (self.feature_moving_mean == True) and (self.feature_moving_var == False):
            feature_mean = self.moving_Fmean
            _, feature_var = tf.nn.moments(inputs, axes = [-1], keepdims=True)
            feature_std = tf.math.sqrt(tf.add(feature_var, self.offset))

        elif (self.feature_moving_mean == False) and (self.feature_moving_var == True):
            feature_mean, _ = tf.nn.moments(inputs, axes = [-1], keepdims=True)
            feature_std = tf.math.sqrt(tf.add(self.moving_Fvar, self.offset))


               
        x_f =   (inputs - feature_mean) / feature_std 
        x_b = (inputs - batch_mean) / batch_std
        numerator =  (x_f * ((1 / self.batch_size) - .0001)) + (x_b * (1- ((1 / self.batch_size) + .0001)))
        x_beta  =  numerator / tf.math.sqrt(self.dk) 

        output =   (self.gamma1 * (x_beta))+ self.beta1
        
        return output
        
        
    def reset_states(self):
        self.moving_Bmean.assign(self.init_mBm)
        self.moving_Bvar.assign(self.init_mBv)
        
        self.moving_Fmean.assign(self.init_mFm)
        self.moving_Fvar.assign(self.init_mFv)

    def call(self, inputs, training):    
        
        return tf.cond(tf.equal(training, True, name='train'),
                       lambda: self.bn_training(inputs), lambda: self.bn_inference(inputs),
                       name = 'call_func') 

    def get_config(self):
        config = super(bln_layer, self).get_config()
        config.update({'stateful': self.stateful})
        config.update({'self.batchsize': self.batchsize})
        config.update({'batch_moving_mean': self.batch_moving_mean})
        config.update({'batch_moving_var': self.batch_moving_var})
        config.update({'feature_moving_mean': self.feature_moving_mean})
        config.update({'feature_moving_var': self.feature_moving_var})
        
        
        return config


#####################################################################################################################   
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
