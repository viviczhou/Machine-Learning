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
    When present, do NOT use development data. When this argument isn't provided, the development data is used to control         training.

    --weights_files [hidden_weights_filename] [output_weights_filename]
    When present, initialize the two matrices with the values present in these files. 

    --hidden_dim [int]
    Only provided if --weights_files is NOT provided. Specifies the number of nodes in the network's single hidden layer.         **NOTE:** *the autograder will never use this. This argument is provided to you as a convenience.*

    --print_weights
    When present, print the two weight matrices. The skeleton file does this for you.

    --train_file [filename]
    Load training data from here (default is /u/cs246/data/adult/a7a.train).

    --dev_file [filename]
    Load dev data from here (default is /u/cs246/data/adult/a7a.dev).

    --test_file [filename]
    Load test data from here (default is /u/cs246/data/adult/a7a.test).
