# Batch Layer Normalization
# A new normalization layer for CNNs and RNNs

This study introduces a new normalization layer termed Batch Layer Normalization
(BLN) to reduce the problem of internal covariate shift in deep learning layers.
BLN as a combined version of batch and layer normalization adaptively puts
appropriate weight on mini-batch and feature normalization based on the inverse
size of mini-batches to normalize the input to a layer during the learning process.
It also performs the exact computation at training and inference times with a
minor change, using mini-batch statistics or population statistics. The decision
process to use statistics of mini-batch or population gives BLN the ability to play a
comprehensive role in the hyperparameter optimization process of models. The
key advantage of BLN is that it does not harm the theoretical analysis of being
independent from the input data and is heavily dependent on the amount of training
data, the task we are performing, and the size of batches. Test results indicate the
application potential of BLN and its faster convergence than batch normalization
and layer normalization in both Convolutional and Recurrent Neural Networks.
# Dependencies
This code requires the following:

    Python 3.5 or greater
    TensorFlow 2.0 or greater

 
