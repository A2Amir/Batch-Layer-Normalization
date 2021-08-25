#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K
import numpy as np
import os



######################################################################################################################   

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


######################################################################################################################   
    
class custombn_paper(tf.keras.layers.Layer):
    
    """
    bn_paper : https://arxiv.org/abs/1502.03167v3
    
    Not implementing momentum as defined in the keras BatchNorm layer
    
    Population mean and variance are calculated as defined in the paper
    
    This process can only be used when mini-batch size > 1
    
    """
    
    
    def __init__(self, stateful, batchsize, **kwargs):
        
        super(custombn_paper, self).__init__(**kwargs)
        
        self.stateful = stateful
        self.batch_size = tf.cast(batchsize, 'float32')
      

    def build(self, input_shape):
        
        if len(input_shape) == 2:
            bn_shape = (1, input_shape[1])
           
        elif len(input_shape) == 3:
            bn_shape = (1,input_shape[1], input_shape[2])
        elif len(input_shape) == 4:
            bn_shape = (1,input_shape[1], input_shape[2], input_shape[3])
        else:
            print('layer shape must be 2D or 3D or 4D')
        
        
        self.gamma = self.add_weight(name = 'scale', shape = (1, input_shape[-1]), initializer = tf.keras.initializers.ones(),
                                    trainable = True)
        self.beta = self.add_weight(name = 'shift', shape = (1,input_shape[-1]), initializer = tf.keras.initializers.zeros(),
                                   trainable = True)
        self.offset = tf.Variable(0.001, dtype = 'float32', trainable=False)
        
        self.moving_mean = self.add_weight(name = 'moving_mean', shape = bn_shape,
                                           initializer = tf.keras.initializers.Zeros(),
                                          trainable = False)
        
        self.moving_var =  self.add_weight(name = 'moving_var', shape = bn_shape,
                                           initializer = tf.keras.initializers.Zeros(),
                                          trainable = False)
        
        self.batch_count = tf.Variable(0, dtype = 'float32', name = 'batchcount', trainable=False)
    
        
        self.init_mm = self.moving_mean.read_value()
        self.init_mv = self.moving_var.read_value()


    def bn_training(self, inputs, axes = [0]):

        self.batch_mean, self.batch_var = tf.nn.moments(inputs, axes = axes, keepdims=True)
        
        self.moving_mean.assign_add(self.batch_mean)
        self.moving_var.assign_add(self.batch_var)

        return tf.add(tf.multiply(tf.divide(tf.subtract(inputs, self.batch_mean), 
                                     tf.math.sqrt(tf.add(self.batch_var, self.offset))), self.gamma), self.beta)
    

    
    def update_mm_mv(self):
        """
        Updating mm and mv at the end of epoch
        """        
        self.moving_mean.assign(tf.cond(tf.greater(self.batch_count,0), 
                                        lambda: tf.divide(self.moving_mean,self.batch_count),
                                        lambda: self.moving_mean, name='update_mm'))
        
        self.moving_var.assign(tf.cond(tf.greater(self.batch_count,0), 
                                       lambda: tf.multiply(self.moving_var,
                                                           tf.divide(self.batch_size,
                                                                     tf.multiply(tf.subtract(self.batch_size,1), 
                                                                                 self.batch_count))),
                                       
                                       lambda: self.moving_var, name='update_mv'))
        
    
    def bn_inference(self, inputs):
        
        return tf.add(tf.multiply(tf.divide(tf.subtract(inputs, self.moving_mean), 
                                     tf.math.sqrt(tf.add(self.moving_var, self.offset))), self.gamma), self.beta)
                
    def reset_states(self):
        self.moving_mean.assign(self.init_mm)
        self.moving_var.assign(self.init_mv)

    def call(self, inputs, training):       
        
        return tf.cond(tf.equal(training, True, name='train_or_eval'),
                       lambda: self.bn_training(inputs), lambda: self.bn_inference(inputs),
                      name = 'call_func') 

    def get_config(self):
        
        config = super(custombn_paper, self).get_config()
        config.update({'stateful': self.stateful})
        
        return config

######################################################################################################################
class bn_keras(tf.keras.layers.Layer):
    
    """
    Implementing momentum as defined in the keras BatchNorm layer to calculate
    population mean and variance 
    
    """
    
    
    def __init__(self, momentum, stateful, **kwargs):
        
        super(bn_keras, self).__init__(**kwargs)
        
        self.momentum = momentum
        self.stateful = stateful

        
    def build(self, input_shape):
        shape = input_shape[-1:]
        
        if len(input_shape) == 2:
            bn_shape = (1, input_shape[1])
           
        elif len(input_shape) == 3:
            bn_shape = (1,input_shape[1], input_shape[2])
        elif len(input_shape) == 4:
            bn_shape = (1,input_shape[1], input_shape[2], input_shape[3])
        else:
            print('layer shape must be 2D or 3D or 4D')
        

        
        self.gamma = self.add_weight(name = 'scale', shape =shape,
                                     initializer = tf.keras.initializers.ones(),
                                     #constraint = lambda t:tf.clip_by_value(t,-1,1),
                                     trainable = True)

        
        self.beta = self.add_weight(name = 'shift', shape = shape,
                                    initializer = tf.keras.initializers.zeros(),
                                    trainable = True)
        
        
        self.offset = tf.Variable(0.001, dtype = 'float32', trainable=False)
        
        self.moving_mean = self.add_weight(name = 'moving_mean', shape = bn_shape,
                                           initializer = tf.keras.initializers.Zeros(),
                                          trainable = False)
        
        self.moving_var =  self.add_weight(name = 'moving_var', shape = bn_shape,
                                           initializer = tf.keras.initializers.Zeros(),
                                          trainable = False)

        self.init_mm = self.moving_mean.read_value()
        self.init_mv = self.moving_var.read_value()
        

    def bn_training(self, inputs, axes = [0]):
        
        
        self.batch_mean, self.batch_var = tf.nn.moments(inputs, axes = axes, keepdims=True)
        self.batch_std = K.sqrt(self.batch_var + self.offset)

        # as implemented in tensorflow        
        self.moving_mean.assign((1-self.momentum)*self.moving_mean + self.momentum*self.batch_mean)
        self.moving_var.assign((1-self.momentum)*self.moving_var + self.momentum*self.batch_var)
        
        
        return tf.add(tf.multiply(tf.divide(tf.subtract(inputs, self.batch_mean), 
                                     tf.math.sqrt(tf.add(self.batch_var, self.offset))), self.gamma), self.beta)


    
    def bn_inference(self, inputs):        
        

        return tf.add(tf.multiply(tf.divide(tf.subtract(inputs, self.moving_mean), 
                                     tf.math.sqrt(tf.add(self.moving_var, self.offset))), self.gamma), self.beta)
        
        
    def reset_states(self):
        
        self.moving_mean.assign(self.init_mm)
        self.moving_var.assign(self.init_mv)
        
        
    def call(self, inputs, training):     
        
        return tf.cond(tf.equal(training, True),lambda: self.bn_training(inputs), lambda: self.bn_inference(inputs))       
    
    def get_config(self):
        
        config = super(bn_keras, self).get_config()
        config.update({'momentum': self.momentum})
        
        return config
            
 ########################################################################################################

class custom_BLN_Layer(tf.keras.layers.Layer):   
    """
    This layer implements new equation for normalizing Features  and Batches

    """
    
    def __init__(self, stateful, batchsize,
                 batch_moving_mean=True, batch_moving_var=True,
                 feature_moving_mean=False, feature_moving_var=False,
                 **kwargs):
        
        super(custom_BLN_Layer, self).__init__(**kwargs)
        self.stateful = stateful
        self.batchsize = batchsize
        self.batch_size = tf.cast(self.batchsize, 'float32')
        
        self.batch_moving_mean = batch_moving_mean
        self.batch_moving_var = batch_moving_var
        
        self.feature_moving_mean = feature_moving_mean
        self.feature_moving_var = feature_moving_var

    def build(self, input_shape):
        
        shape = input_shape[-1:]
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

        '''self.gamma2 = self.add_weight(name = 'scale2', shape = shape,
                                     initializer = tf.keras.initializers.ones(),
                                    trainable = True)
        
        self.beta2 = self.add_weight(name = 'shift2', shape = shape,
                                    initializer = tf.keras.initializers.zeros(),
                                   trainable = True)
       '''
        
        self.gamma3 = self.add_weight(name = 'scale3', shape = shape,
                                     initializer = tf.keras.initializers.ones(),
                                    trainable = True)
        
        self.beta3 = self.add_weight(name = 'shift3', shape = shape,
                                    initializer = tf.keras.initializers.zeros(),
                                   trainable = True)
        
        self.offset = tf.Variable(0.001, dtype = 'float32', trainable=False)
        self.landa = tf.Variable(0.001, dtype = 'float32', trainable=True, name='landa',
                                 constraint = lambda t:tf.clip_by_value(t,0,1))
        
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
        
   
        output1 =   (((self.gamma1/ feature_std) * inputs) - ((self.gamma1/ feature_std) * (feature_mean + batch_mean )))+ self.beta1 #
        output2 =   (((self.gamma1/ batch_std) * inputs) - ((self.gamma1/batch_std) * ( feature_mean + batch_mean)))+ self.beta1     
       
        
        #output1 = tf.add(tf.multiply(tf.divide(tf.subtract(inputs, batch_mean),batch_std) , self.gamma1), self.beta1)
        #output2 = tf.add(tf.multiply(tf.divide(tf.subtract(inputs, feature_mean),feature_std) , self.gamma1), self.beta1)

        output =  (output1 * self.landa) + ( output2 * (1 - self.landa)  ) 
        output =   (self.gamma3 * (output))+ self.beta3   
        
        return output
    

    
    def update_mm_mv(self):
        """
        Updating mBm and mBv, mFm and mFv at the end of epoch
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

        output1 =   (((self.gamma1/  feature_std ) * inputs)) - ((self.gamma1/  feature_std) *  (feature_mean + batch_mean ))+ self.beta1
        output2 =   (((self.gamma1/ batch_std  ) * inputs)) - ((self.gamma1/ batch_std) * ( feature_mean + batch_mean) )+ self.beta1
        

        #output1 = tf.add(tf.multiply(tf.divide(tf.subtract(inputs, batch_mean),batch_std) , self.gamma1), self.beta1)
        #output2 = tf.add(tf.multiply(tf.divide(tf.subtract(inputs, feature_mean),feature_std) , self.gamma1), self.beta1)
        
        output =  (output1 * self.landa) + (output2 * (1 - self.landa)) 
        output =   (self.gamma3 * output )+ self.beta3   
        
        return output
        
        
    def reset_states(self):
        self.moving_Bmean.assign(self.init_mBm)
        self.moving_Bvar.assign(self.init_mBv)
        
        self.moving_Fmean.assign(self.init_mFm)
        self.moving_Fvar.assign(self.init_mFv)

    def call(self, inputs, training):    
        
        return tf.cond(tf.equal(training, True, name='train_or_eval'),
                       lambda: self.bn_training(inputs), lambda: self.bn_inference(inputs),
                       name = 'call_func') 

    def get_config(self):
        config = super(custom_BLN_Layer, self).get_config()
        config.update({'stateful': self.stateful})
        config.update({'self.batchsize': self.batchsize})
        config.update({'batch_moving_mean': self.batch_moving_mean})
        config.update({'batch_moving_var': self.batch_moving_var})
        config.update({'feature_moving_mean': self.feature_moving_mean})
        config.update({'feature_moving_var': self.feature_moving_var})
        
        
        return config
              