#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import os

#####################################################################################################################
class custombn_paper_callback(tf.keras.callbacks.Callback):
    
    """
    bn_paper : https://arxiv.org/abs/1502.03167v3
    This callback resets the moving mean and variances at the end of each epoch.
    """
    
    def __init__(self, batchsize, **kwargs):
        
        super(custombn_paper_callback, self).__init__(**kwargs)
                
        self.batchsize = batchsize
        self.batchcount = tf.Variable(0, dtype = tf.int32, trainable=False)
        
        
    def on_train_begin(self, logs =None):
        """
        At the beginning of the training, the batchsize is assigned to
        the batchsize variables of the bn layers for correct calc. of 
        the mm/mv.        
        """
        for layer in self.model.layers:
            if layer.__class__.__name__ == 'custombn_paper':
                layer.batchsize.assign(self.batchsize)      
        
    
    
    def on_train_batch_end(self, batch, logs=None):
        """
        This function in combination with functions below
        is used to pass the total number of batches in the training data
        for the calculation of mm/mv.
        """
        self.batchcount.assign(batch)
        tf.cond(tf.equal(self.batchcount,1), self.at_batch_one, self.at_batch_not_one)     
        
    def at_batch_one(self):
        """
        When the first batch of the training data has been processed 
        this function retrieves the total batches(steps) and updates
        the corresponding variable in the batch norm layers for 
        calculating moving mean and moving var.                  
        """
        
        for layer in self.model.layers:
            if layer.__class__.__name__ == 'custombn_paper':
                layer.batch_count.assign(self.model.history.params['steps'])

        return None
        
    def at_batch_not_one(self):
        tf.cond(tf.equal(self.batchcount, self.model.history.params['steps']-1), self.at_last_batch, self.at_any_batch)
    
    def at_last_batch(self):
        """
        This layer updates mm/mv after the last batch of the 
        training data has been processed.        
        """
        for layer in self.model.layers:
            if hasattr(layer, 'update_mm_mv'):
                layer.update_mm_mv()
        
        return None
    
    def at_any_batch(self):
        """
        Nothing to do any other batch number
        """
        return None

       
    def on_epoch_begin(self, epoch, logs = None):
        """
        At the beginning of every epoch, the mm/mv need to be reset to zero.
        """
        self.model.reset_states()    
        
######################################################################################################################
class bn_keras_callbck(tf.keras.callbacks.Callback):
    """
    For resetting mm/mv after each epoch.
    """
    
    def __init__(self, **kwargs):
        
        super(bn_keras_callbck, self).__init__(**kwargs)
            
    def on_epoch_begin(self, epoch, logs = None):      
        self.model.reset_states()

######################################################################################################################
class custom_BLNLayer_callback(tf.keras.callbacks.Callback):
    """
    nothing to do
    """
    
    def __init__(self, **kwargs):
        
        super(custom_BLNLayer_callback, self).__init__(**kwargs)
            
    def on_epoch_begin(self, epoch, logs = None):
        print(epoch)
    
 ######################################################################################################################

class comb_cBNpaper_cBLNLayer_callback(tf.keras.callbacks.Callback):
    
    """
    This callback is a callback for the combination of the custom_BLNormalization (custom Batch and Layer  Normalization) and
    custombn_paper (custom Batch Normalization Paper) approches.
    
    This callback resets the moving mean and variances at the end of each epoch.
    """
    
    def __init__(self, batchsize, **kwargs):
        
        super(comb_cBNpaper_cBLNLayer_callback, self).__init__(**kwargs)
        
        self.batchsize = batchsize                
        self.batchcount = tf.Variable(0, dtype = tf.int32, trainable=False)

    def on_train_begin(self, logs =None):
        """
        At the beginning of the training, the batchsize is assigned to
        the batchsize variables of the bn layers for correct calc. of 
        the mm/mv.        
        """
        for layer in self.model.layers:
            if layer.__class__.__name__ == 'comb_cBNpaper_cBLNLayer':
                layer.batchsize.assign(self.batchsize)      

    def on_train_batch_end(self, batch, logs=None):
        """
        This function in combination with functions below
        is used to pass the total number of batches in the training data
        for the calculation of mm/mv.
        """
        self.batchcount.assign(batch)
        tf.cond(tf.equal(self.batchcount,1), self.at_batch_one, self.at_batch_not_one)     
        
    def at_batch_one(self):
        """
        When the first batch of the training data has been processed 
        this function retrieves the total batches(steps) and updates
        the corresponding variable in the batch norm layers for 
        calculating moving mean and moving var.                  
        """
        
        for layer in self.model.layers:
            if layer.__class__.__name__ == 'comb_cBNpaper_cBLNLayer':
                layer.batch_count.assign(self.model.history.params['steps'])

        return None
        
    def at_batch_not_one(self):
        tf.cond(tf.equal(self.batchcount, self.model.history.params['steps']-1), self.at_last_batch, self.at_any_batch)
    
    def at_last_batch(self):
        """
        This layer updates mm/mv after the last batch of the 
        training data has been processed.        
        """
        for layer in self.model.layers:
            if hasattr(layer, 'update_mm_mv'):
                layer.update_mm_mv()
        
        return None
    
    def at_any_batch(self):
        """
        Nothing to do any other batch number
        """
        return None

       
    def on_epoch_begin(self, epoch, logs = None):
        """
        At the beginning of every epoch, the mm/mv need to be reset to zero.
        """
        
        self.model.reset_states()       


######################################################################################################################
class comb_cBNpaper_cBLNLayer_chMean_callback(tf.keras.callbacks.Callback):
    
    """
    
    This callback is a callback for the combination of the custom_BLNormalization (custom Batch and Layer  Normalization) and
    custombn_paper (custom Batch Normalization Paper) approches plus adding the channle mean and variance.
    
    
    This callback resets the moving mean and variances at the end of each epoch.
    """
    
    def __init__(self, batchsize, **kwargs):
        
        super(comb_cBNpaper_cBLNLayer_chMean_callback, self).__init__(**kwargs)
        
        self.batchsize = batchsize                
        self.batchcount = tf.Variable(0, dtype = tf.int32, trainable=False)

    def on_train_begin(self, logs =None):
        """
        At the beginning of the training, the batchsize is assigned to
        the batchsize variables of the bn layers for correct calc. of 
        the mm/mv.        
        """
        for layer in self.model.layers:
            if layer.__class__.__name__ == 'comb_cBNpaper_cBLNLayer_chMean':
                layer.batchsize.assign(self.batchsize)      

    def on_train_batch_end(self, batch, logs=None):
        """
        This function in combination with functions below
        is used to pass the total number of batches in the training data
        for the calculation of mm/mv.
        """
        self.batchcount.assign(batch)
        tf.cond(tf.equal(self.batchcount,1), self.at_batch_one, self.at_batch_not_one)     
        
    def at_batch_one(self):
        """
        When the first batch of the training data has been processed 
        this function retrieves the total batches(steps) and updates
        the corresponding variable in the batch norm layers for 
        calculating moving mean and moving var.                  
        """
        
        for layer in self.model.layers:
            if layer.__class__.__name__ == 'comb_cBNpaper_cBLNLayer_chMean':
                layer.batch_count.assign(self.model.history.params['steps'])

        return None
        
    def at_batch_not_one(self):
        tf.cond(tf.equal(self.batchcount, self.model.history.params['steps']-1), self.at_last_batch, self.at_any_batch)
    
    def at_last_batch(self):
        """
        This layer updates mm/mv after the last batch of the 
        training data has been processed.        
        """
        for layer in self.model.layers:
            if hasattr(layer, 'update_mm_mv'):
                layer.update_mm_mv()
        
        return None
    
    def at_any_batch(self):
        """
        Nothing to do any other batch number
        """
        return None

       
    def on_epoch_begin(self, epoch, logs = None):
        """
        At the beginning of every epoch, the mm/mv need to be reset to zero.
        """
        
        self.model.reset_states()       


######################################################################################################################

class ln_keras_callbck(tf.keras.callbacks.Callback):
    """
    nothing to do
    """
    def __init__(self, **kwargs):
        
        super(ln_keras_callbck, self).__init__(**kwargs)
            
    def on_epoch_begin(self, epoch, logs = None):
        print(epoch)

######################################################################################################################

# Callback for saving best model
def save_best_model_callback(model_name):
    path = './models/'+ model_name
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)
        
    savemodel_cb = tf.keras.callbacks.ModelCheckpoint(os.path.join(path,'model_{epoch}-{val_loss:.3f}.h5'),
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
                                                    histogram_freq=1,
                                                    write_graph=True, write_images=True,
                                                   profile_batch = 100000000)
    return tensorboard_cb

#####################################################################################################################

def create_callback_list(savemodel_cb, tensorboard_cb, custom_cb):
    cb_list = []
    cb_list.append([tensorboard_cb, savemodel_cb, custom_cb]) 
    cb_list = cb_list[0]
    return cb_list