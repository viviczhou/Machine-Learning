# Machine-Learning
Datasets used for algorithms are included in DATA file.

## Perceptron Algorithm
This file including the python code that implemented perceptron algorithm for the adult income dataset and the results of experiment with performance as a function of number of iterations and the use of dev set.

### Interface:

    --iterations [int]
    The program should stop after this many iterations through the training data.

    --lr [float]
    The program should use this as the learning rate.

    --nodev
    When this is provided, the program should NOT make any use of development data. 
    This argument will only be provided when both --iterations and --lr are also provided.

    --train_file [filename]
    Load training data from here (default is /u/cs446/data/adult/a7a.train).

    --dev_file [filename]
    Load dev data from here (default is /u/cs446/data/adult/a7a.dev).

    --test_file [filename]
    Load test data from here (default is /u/cs446/data/adult/a7a.test).

## Backprop
This file including the python code that implemented backpropagation algorithm for the adult income dataset and the results of experiment with performance as a function of number of iterations and the use of dev set.

### Details:
    -For the loss function, use log likelihood as shown in class.
    -Use a batch size of 1 (i.e. update the weights after every data point).
    -Use a network with only 1 hidden layer (of variable size), and a single-node output layer.
    -Use sigmoid as the activation for both the hidden layer and output layer.
    -There is a bias at every layer.
    
### Interface:

    --iterations [int]
    Stop training after this many iterations through the data (sometimes called epochs in the literature).

    --lr [float]
    Learning rate.

    --nodev
    When present, do NOT use development data. When this argument isn't provided, the development data is used to control training.

    --weights_files [hidden_weights_filename] [output_weights_filename]
    When present, initialize the two matrices with the values present in these files. 

    --hidden_dim [int]
    Only provided if --weights_files is NOT provided. Specifies the number of nodes in the network's single hidden layer. NOTE: the autograder will never use this. This argument is provided to you as a convenience.*

    --print_weights
    When present, print the two weight matrices. The skeleton file does this for you.

    --train_file [filename]
    Load training data from here (default is /u/cs246/data/adult/a7a.train).

    --dev_file [filename]
    Load dev data from here (default is /u/cs246/data/adult/a7a.dev).

    --test_file [filename]
    Load test data from here (default is /u/cs246/data/adult/a7a.test).

## EM Algorithm Mixture of Gaussians
This file including the python code that implemented EM fitting of a mixture of gaussians on the two-dimensional data set points.dat and the results of experiment with performance as a function of number of iterations, numbers of mixtures, various types of covariance matrices, and the use of dev set.

### Interface

    --iterations [int]
    Stop training after this many iterations through the data.

    --nodev
    When present, do NOT use development data. When this argument isn't provided, the development data is used to control training.

    --clusters_file [filename]
    When present, initializes clusters (either Gaussian or aspect model) from the given file. Not allowed to be provided with --cluster_num (see below).

    --cluster_num [int]
    Only provided if --clusters_file is NOT provided. Specifies the number of clusters (hidden variables; K in the notes).  
    
    --print_params
    When present, parameters of the clusters will be printed. 

    --data_file [filename]
    Load data from here. The last 10% will be reserved as dev data (and discarded if --nodev is provided).
    
## EM Algorithm HMM
This file including the python code that implemented EM to train an HMM for the two-dimensional data set points.dat and the results of experiment with performance as a function of number of iterations, numbers of mixtures, various types of covariance matrices, and the use of dev set. The first 900 obervations are used as a single training sequence, and the last 100 are used as a single development sequence.

The interface of this file is the same as the previous one.
