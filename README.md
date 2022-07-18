




    
 <h1  style="display: inline-block;" align = "center"> Batch Layer Normalization</h1>
<h1 align = "center"> A new normalization layer for CNNs and RNNs </h1>
  
This study introduces <b>a new normalization layer termed Batch Layer Normalization (BLN)</b> to reduce the problem of internal covariate
shift in deep neural network layers. As a combined version of batch and layer normalization, BLN adaptively puts appropriate weight
on mini-batch and feature normalization based on the inverse size of mini-batches to normalize the input to a layer during the learning
process. It also performs the exact computation with a minor change at inference times, using either mini-batch statistics or population
statistics. The decision process to use statistics of mini-batch or population gives BLN the ability to play a comprehensive role in
the hyper-parameter optimization process of models. The key advantage of BLN is that it does not harm the theoretical analysis of
being independent of the input data, and its statistical configuration heavily depends on the amount of training data, the task we
are performing, and the size of batches. Test results indicate the application potential of BLN and its faster convergence than batch
normalization and layer normalization in both Convolutional and Recurrent Neural Networks. 


# Dependencies

In order to use the helper codes, create the environment from the environment.yml file  

        conda env create -f environment.yml

Activate the new environment: 

        conda activate bln
        
Verify that the new environment was installed correctly

        conda env list
        
