 <h1  style="display: inline-block;" align = "center">
   
   [Batch Layer Normalization, A new normalization layer for CNNs and RNNs](https://arxiv.org/abs/2209.08898)

 </h1>
  
This study introduces a new normalization layer termed Batch Layer Normalization (BLN) to reduce the problem of internal covariate shift in deep neural network layers. As a combined version of batch and layer normalization, BLN adaptively puts appropriate weight on mini-batch and feature normalization based on the inverse size of mini-batches to normalize the input to a layer during the learning process. It also performs the exact computation with a minor change at inference times, using either mini-batch statistics or population statistics. The decision process to either use statistics of mini-batch or population gives BLN the ability to play a comprehensive role in the hyper-parameter optimization process of models. The key advantage of BLN is the support of the theoretical analysis of being independent of the input data, and its statistical configuration heavily depends on the task performed, the amount of training data, and the size of batches. Test results indicate the application potential of BLN and its faster convergence than batch normalization and layer normalization in both Convolutional and Recurrent Neural Networks. 


# Dependencies

In order to use the helper codes, create the environment from the environment.yml file  

        conda env create -f environment.yml

Activate the new environment: 

        conda activate bln
        
Verify that the new environment was installed correctly

        conda env list
        
# How to use

```python
from helpers.bln_layer import  bln_layer
from helpers.bln_callback import bln_callback 

# as a normalization layer
x = bln_layer(stateful = True, batchsize= batch_size, name = 'bn1', 
              batch_moving_mean = False, batch_moving_var = False,
              feature_moving_mean = False, feature_moving_var = False)(x) 
    


# add Callback for resetting moving means and variances at the end of each epoch
model_history =  model.fit(train_dataset, epochs = epochs, verbose = 1,
                           validation_data = valid_dataset, callbacks = [bln_callback()], shuffle = True)
                           
                           
                           

```
Use a grid-search algorithm or other hyper-parameter tuning techniques to find the best configuration of statistics <b>(batch_moving_mean, batch_moving_var, feature_moving_mean, feature_moving_var)</b> among the possible configurations with lower loss and higher accuracy.


```
            batch_moving_mean = {True or False}, batch_moving_var = {True or False},  

            feature_moving_mean = {True or False},  feature_moving_var = {True or False} 
```   
Use the best configuration for the rest of the training or fine-tuning and network testing.

* For more infromation see [the example](https://github.com/A2Amir/Batch-Layer-Normalization/blob/main/Cifar10%20(With%20the%20whole%20training%20set%20and%20batch%20size%2025).ipynb).
