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

def reset_graph():
    tf.keras.backend.clear_session()
    print('session is clear')
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
                                              trainable = self.trainable)
        
        self.bias = self.add_weight(name = 'bias', shape = (self.units,), dtype = self.dtype,
                                   initializer=tf.keras.initializers.zeros(), trainable = self.trainable)
        
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
    
    
    def __init__(self, stateful, **kwargs):
        
        super(custombn_paper, self).__init__(**kwargs)
        
        self.stateful = stateful
      

    def build(self, input_shape):
        self.gamma = self.add_weight(name = 'scale', shape = (1, input_shape[-1]), initializer = tf.keras.initializers.ones(),
                                    trainable = self.trainable)
        self.beta = self.add_weight(name = 'shift', shape = (1,input_shape[-1]), initializer = tf.keras.initializers.zeros(),
                                   trainable = self.trainable)
        self.offset = tf.Variable(0.001, dtype = 'float32', trainable=False)
        
        self.moving_mean = self.add_weight(name = 'moving_mean', shape = (1, input_shape[-1]),
                                           initializer = tf.keras.initializers.Zeros(),
                                          trainable = False)
        
        self.moving_var =  self.add_weight(name = 'moving_var', shape = (1, input_shape[-1]),
                                           initializer = tf.keras.initializers.Zeros(),
                                          trainable = False)
        
        self.batch_count = tf.Variable(0, dtype = 'float32', name = 'batchcount', trainable=False)
        
        self.batchsize = tf.Variable(2, dtype = 'float32', name='batchsize', trainable=False)

        
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
                                                           tf.divide(self.batchsize,
                                                                     tf.multiply(tf.subtract(self.batchsize,1), 
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

        
        self.gamma = self.add_weight(name = 'scale', shape =shape,
                                     initializer = tf.keras.initializers.ones(),
                                     #constraint = lambda t:tf.clip_by_value(t,-1,1),
                                     trainable = True)

        
        self.beta = self.add_weight(name = 'shift', shape = shape,
                                    initializer = tf.keras.initializers.zeros(),
                                    trainable = True)
        

        
        self.offset = tf.Variable(0.001, dtype = 'float32', trainable=False)
        
        self.moving_mean = self.add_weight(name = 'moving_mean', shape = (1, input_shape[-1]),
                                           initializer = tf.keras.initializers.Zeros(),
                                          trainable = False)
        
        self.moving_var =  self.add_weight(name = 'moving_var', shape = (1, input_shape[-1]),
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


class customBatchLayerNormalLayer(tf.keras.layers.Layer):
    """
    This layer implements new equation for normalizing features  

    """

    def __init__(self, **kwargs):
        super(customBatchLayerNormalLayer, self).__init__(**kwargs)
        self.gamma1 = None
        self.gamma2 = None
        self.beta1 = None
        self.beta2 = None
        self.offset = None


    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        shape = input_shape[-1:]
                
        self.gamma1 = self.add_weight(name = 'scale1', shape =shape,
                                     initializer = tf.keras.initializers.ones(),
                                     #constraint = lambda t:tf.clip_by_value(t,-1,1),
                                     trainable = True)
        
        self.gamma2 = self.add_weight(name = 'scale2', shape =shape,
                                     initializer = tf.keras.initializers.ones(),
                                     # constraint = lambda t:tf.clip_by_value(t,-1,1),
                                     trainable = True)
        
        self.beta1 = self.add_weight(name = 'shift1', shape = shape,
                                    initializer = tf.keras.initializers.zeros(),
                                    trainable = True) 
        
        self.beta2 = self.add_weight(name = 'shift2', shape = shape,
                                    initializer = tf.keras.initializers.zeros(),
                                    trainable = True)
        
        self.gamma3 = self.add_weight(name = 'scale3', shape =shape,
                                     initializer = tf.keras.initializers.ones(),
                                     # constraint = lambda t:tf.clip_by_value(t,-1,1),
                                     trainable = True)
        
        self.beta3 = self.add_weight(name = 'shift3', shape = shape,
                                    initializer = tf.keras.initializers.zeros(),
                                    trainable = True) 
        self.landa = tf.Variable(0.5, dtype = 'float32', trainable=True, name='landa')

        self.offset = tf.Variable(0.001, dtype = 'float32', trainable=False)

        super(customBatchLayerNormalLayer, self).build(input_shape)
        
    def bn_training(self, inputs):
        
        ch_mean, ch_variance = tf.nn.moments(inputs, axes =[-1], keepdims=True)
        ch_std = K.sqrt(ch_variance + self.offset )
        
        batch_mean, batch_var = tf.nn.moments(inputs, axes =[0], keepdims=True)
        batch_std = K.sqrt(batch_var + self.offset )
         
        output1 =   (((self.gamma1/ch_std) * inputs) - ((self.gamma1/ch_std) * ch_mean))+ self.beta1
        output2 =   (((self.gamma2/batch_std) * inputs) - ((self.gamma2/batch_std) * batch_mean))+ self.beta2
        output =     self.gamma3 * ((self.landa * output1) + ((1-self.landa )* output2)) + self.beta3
        
        return   output
    
    def bn_inference(self, inputs):

        ch_mean, ch_variance = tf.nn.moments(inputs, axes =[-1], keepdims=True)
        ch_std = K.sqrt(ch_variance + self.offset )
        
        batch_mean, batch_var = tf.nn.moments(inputs, axes =[0], keepdims=True)
        batch_std = K.sqrt(batch_var + self.offset )
         
        output1 =   (((self.gamma1/ch_std) * inputs) - ((self.gamma1/ch_std) * ch_mean))+ self.beta1
        output2 =   (((self.gamma2/batch_std) * inputs) - ((self.gamma2/batch_std) * batch_mean))+ self.beta2
        output =     self.gamma3 *((self.landa * output1) + ((1-self.landa )* output2)) + self.beta3

        return output
    
    def call(self, inputs, training):       
        
        return tf.cond(tf.equal(training, True, name='train_or_eval'),
                       lambda: self.bn_training(inputs),
                       lambda: self.bn_inference(inputs), name = 'call_func') 
    
    
    
########################################################################################################

class comb_cBNpaper_cBLNLayer(tf.keras.layers.Layer):
    
    """
    This layer implements a combined appraoch of the customBatchLayerNormalization (custom Batch and Layer  Normalization) and
    custombn_paper (custom Batch Normalization Paper) approches.
    
    """
    
    
    def __init__(self, stateful, **kwargs):
        
        super(comb_cBNpaper_cBLNLayer, self).__init__(**kwargs)       
        self.stateful = stateful

    def build(self, input_shape):
        
        shape = input_shape[-1:]
        if len(input_shape) == 2:
            bn_shape = (1,input_shape[1])
            
        elif len(input_shape) == 3:
            bn_shape = (1,input_shape[1], input_shape[2])
        else:
            print('layer shape must be 2D or 3D')

        self.gamma1 = self.add_weight(name = 'scale1', shape = shape,
                                     initializer = tf.keras.initializers.ones(),
                                     trainable = self.trainable)
       
        self.beta1 = self.add_weight(name = 'shift1', shape = shape,
                                    initializer = tf.keras.initializers.zeros(),
                                    trainable = self.trainable)

        self.gamma2 = self.add_weight(name = 'scale2', shape = shape,
                                     initializer = tf.keras.initializers.ones(),
                                    trainable = self.trainable)
        
        self.beta2 = self.add_weight(name = 'shift2', shape = shape,
                                     initializer = tf.keras.initializers.zeros(),
                                     trainable = self.trainable)
        
        self.offset = tf.Variable(0.001, dtype = 'float32', trainable=False)
        
        
        self.moving_mean = self.add_weight(name = 'moving_mean', shape = bn_shape,
                                            initializer = tf.keras.initializers.Zeros(),
                                            trainable = False)
        
        self.moving_var =  self.add_weight(name = 'moving_var', shape = bn_shape ,
                                            initializer = tf.keras.initializers.Zeros(),
                                            trainable = False)

        self.batch_count = tf.Variable(0, dtype = 'float32', name = 'batchcount', trainable=False)
        self.batchsize = tf.Variable(2, dtype = 'float32', name='batchsize', trainable=False)

        self.init_mm = self.moving_mean.read_value()
        self.init_mv = self.moving_var.read_value()
        

    def bn_training(self, inputs, axes = [0]):
        


        self.batch_mean, self.batch_var = tf.nn.moments(inputs, axes = axes, keepdims=True)
        self.batch_std = K.sqrt(self.batch_var + self.offset)

        self.moving_mean.assign_add(self.batch_mean)
        self.moving_var.assign_add(self.batch_var)

        self.ch_mean, self.ch_var = tf.nn.moments(inputs, axes = [-1], keepdims=True)
        self.ch_std = K.sqrt(self.ch_var + self.offset )

        output1 =   (((self.gamma1/ self.ch_std) * inputs) - ((self.gamma1/ self.ch_std) * self.ch_mean))+ self.beta1
        output2 =   (((self.gamma2/self.batch_std) * inputs) - ((self.gamma2/self.batch_std) *  self.batch_mean))+ self.beta2

        output =    (output1 + output2)# + self.offset

        return output
    

    
    def update_mm_mv(self):
        """
        Updating mm and mv at the end of epoch
        """        
        self.moving_mean.assign(tf.cond(tf.greater(self.batch_count,0), 
                                        lambda: tf.divide(self.moving_mean,self.batch_count),
                                        lambda: self.moving_mean, name='update_mm'))
        
        self.moving_var.assign(tf.cond(tf.greater(self.batch_count,0), 
                                       lambda: tf.multiply(self.moving_var, tf.divide(self.batchsize,
                                                                                      tf.multiply(tf.subtract(self.batchsize,1),
                                                                                                  self.batch_count))),
                                      
                                       lambda: self.moving_var, name='update_mv'))
        

    def bn_inference(self, inputs):
        
        
        batch_std = tf.math.sqrt(tf.add(self.moving_var, self.offset))
        batch_mean = self.moving_mean
        
        ch_mean, ch_var = tf.nn.moments(inputs, axes = [-1], keepdims=True)
        ch_std = K.sqrt(ch_var + self.offset )
        
        output1 =   (((self.gamma1/ ch_std) * inputs) - ((self.gamma1/ ch_std) * ch_mean))+ self.beta1
        output2 =   (((self.gamma2/ batch_std) * inputs) - ((self.gamma2/batch_std) *  batch_mean))+ self.beta2
        output =    (output1 + output2) #+ self.offset
        
        return output
        
        
    def reset_states(self):
        self.moving_mean.assign(self.init_mm)
        self.moving_var.assign(self.init_mv)
        
    def call(self, inputs, training):       
        
        return tf.cond(tf.equal(training, True, name='train_or_eval'),lambda: self.bn_training(inputs),
                       lambda: self.bn_inference(inputs), name = 'call_func') 

    def get_config(self):
        
        config = super(comb_cBNpaper_cBLNLayer, self).get_config()
        config.update({'stateful': self.stateful})
        
        return config
        
######################################################################################################     

class comb_cBNpaper_cBLNLayer_chMean(tf.keras.layers.Layer):
    
    """
    This layer implements a combined appraoch of the customBatchLayerNormalization (custom Batch and Layer  Normalization) and
    custombn_paper (custom Batch Normalization Paper) approches plus adding channel mean and variance.
    
    """
    
    
    def __init__(self, stateful, batch_size, **kwargs):
        
        super(comb_cBNpaper_cBLNLayer_chMean, self).__init__(**kwargs)
        self.stateful = stateful
        self.batch_size = batch_size

    def build(self, input_shape):
        
        shape = input_shape[-1:]
        if len(input_shape) == 2:
            bn_shape = (1, input_shape[1])
            ch_shape = (self.batch_size,1)
           
        elif len(input_shape) == 3:
            bn_shape = (1,input_shape[1], input_shape[2])
            ch_shape = (self.batch_size, input_shape[1], 1)
        else:
            print('layer shape must be 2D or 3D')

        self.gamma1 = self.add_weight(name = 'scale1', shape = shape,
                                     initializer = tf.keras.initializers.ones(),
                                    trainable = self.trainable)
       
        self.beta1 = self.add_weight(name = 'shift1', shape = shape,
                                    initializer = tf.keras.initializers.zeros(),
                                   trainable = self.trainable)

        self.gamma2 = self.add_weight(name = 'scale2', shape = shape,
                                     initializer = tf.keras.initializers.ones(),
                                    trainable = self.trainable)
        
        self.beta2 = self.add_weight(name = 'shift2', shape = shape,
                                    initializer = tf.keras.initializers.zeros(),
                                   trainable = self.trainable)
        
        self.offset = tf.Variable(0.001, dtype = 'float32', trainable=False)
        
        
        self.moving_Bmean = self.add_weight(name = 'moving_Bmean', shape = bn_shape,
                                            initializer = tf.keras.initializers.Zeros(),
                                            trainable = False)
        self.moving_Bvar =  self.add_weight(name = 'moving_Bvar', shape = bn_shape ,
                                            initializer = tf.keras.initializers.Zeros(),
                                            trainable = False)
        
        self.moving_Cmean = self.add_weight(name = 'moving_Cmean', shape = ch_shape,
                                            initializer = tf.keras.initializers.Zeros(),
                                            trainable = False)
        self.moving_Cvar =  self.add_weight(name = 'moving_Cvar', shape = ch_shape ,
                                            initializer = tf.keras.initializers.Zeros(),
                                            trainable = False)        
        
        self.batch_count = tf.Variable(0, dtype = 'float32', name = 'batchcount', trainable=False)
        self.batchsize = tf.Variable(2, dtype = 'float32', name='batchsize', trainable=False)
        
        self.init_mBm = self.moving_Bmean.read_value()
        self.init_mBv = self.moving_Bvar.read_value()
        
        self.init_mCm = self.moving_Cmean.read_value()
        self.init_mCv = self.moving_Cvar.read_value()

    def bn_training(self, inputs, axes = [0]):
        
        batch_mean, batch_var = tf.nn.moments(inputs, axes = axes, keepdims=True)
        batch_std = K.sqrt(batch_var + self.offset)
        self.moving_Bmean.assign_add(batch_mean)
        self.moving_Bvar.assign_add(batch_var)
        

        ch_mean, ch_var = tf.nn.moments(inputs, axes = [-1], keepdims=True)
        ch_std = K.sqrt(ch_var + self.offset )
        self.moving_Cmean.assign_add(ch_mean)
        self.moving_Cvar.assign_add(ch_var)
        
   
        output1 =   (((self.gamma1/ ch_std) * inputs) - ((self.gamma1/ ch_std) * ch_mean))+ self.beta1
        output2 =   (((self.gamma2/ batch_std) * inputs) - ((self.gamma2/batch_std) * batch_mean))+ self.beta2     
        output = (output1 + output2) #+ self.beta1#+ self.offset     

        return output
    

    
    def update_mm_mv(self):
        """
        Updating mBm and mBv, mCm and mCv at the end of epoch
        """        
        self.moving_Bmean.assign(tf.cond(tf.greater(self.batch_count,0), 
                                        lambda: tf.divide(self.moving_Bmean,self.batch_count), lambda: self.moving_Bmean,
                                         name='update_mBm'))
        
        self.moving_Bvar.assign(tf.cond(tf.greater(self.batch_count,0), 
                                       lambda: tf.multiply(self.moving_Bvar,
                                                           tf.divide(self.batchsize,
                                                                     tf.multiply(tf.subtract(self.batchsize,1),
                                                                                 self.batch_count))),
                                       lambda: self.moving_Bvar, name='update_mBv'))
        
        self.moving_Cmean.assign(tf.cond(tf.greater(self.batch_count,0), 
                                        lambda: tf.divide(self.moving_Cmean,self.batch_count),
                                        lambda: self.moving_Cmean, name='update_mCm'))
        
        self.moving_Cvar.assign(tf.cond(tf.greater(self.batch_count,0), 
                                       lambda: tf.multiply(self.moving_Cvar,
                                                           tf.divide(self.batchsize,
                                                                     tf.multiply(tf.subtract(self.batchsize,1),
                                                                                 self.batch_count))),
                                       lambda: self.moving_Cvar, name='update_mCv'))
        #tf.print('self.moving_Bmean',self.moving_Bmean)
        #tf.print('self.moving_Cmean',self.moving_Cmean)
    
    def bn_inference(self, inputs):
        
        
        batch_std = tf.math.sqrt(tf.add(self.moving_Bvar, self.offset))
        batch_mean = self.moving_Bmean
        
        ch_std = tf.math.sqrt(tf.add(self.moving_Cvar, self.offset))
        ch_mean = self.moving_Cmean
        
        
        output1 =   (((self.gamma1/ ch_std) * inputs) - ((self.gamma1/ ch_std) * ch_mean))+ self.beta1
        output2 =   (((self.gamma2/batch_std) * inputs) - ((self.gamma2/batch_std) *  batch_mean))+ self.beta2
        output = (output1 + output2) #+ self.offset 
        
        return output
        
        
    def reset_states(self):

        self.moving_Bmean.assign(self.init_mBm)
        self.moving_Bvar.assign(self.init_mBv)
        
        self.moving_Cmean.assign(self.init_mCm)
        self.moving_Cvar.assign(self.init_mCv)

    def call(self, inputs, training):       
        
        return tf.cond(tf.equal(training, True, name='train_or_eval'),
                       lambda: self.bn_training(inputs), lambda: self.bn_inference(inputs),
                      name = 'call_func') 

    def get_config(self):
        
        config = super(comb_cBNpaper_cBLNLayer_chMean, self).get_config()
        config.update({'stateful': self.stateful})
        
        return config
              
        
        
        
        
        
        
        
        
        
        
        
        
        
        