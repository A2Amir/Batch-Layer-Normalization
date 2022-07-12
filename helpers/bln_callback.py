#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import os


######################################################################################################################
class bln_callback(tf.keras.callbacks.Callback):
    
    """
    This callback resets the moving mean and variances at the end of each epoch.
    """
    
    def __init__(self, **kwargs):
        
        super(bln_callback, self).__init__(**kwargs)
        self.batchcount = tf.Variable(0, dtype = tf.int32, trainable=False)

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
            if layer.__class__.__name__ == 'bln_layer':
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