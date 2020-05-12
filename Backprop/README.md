# Backpropagation
## Instructions
### Steps:

0) START EARLY. Plan to have the basic functionality done by Friday, February 1st.
1) Download the archive with the assignment files, and run 'tar xzvf backprop_files.tar.gz'. Rename the skeleton Python 3 file to LastName_backprop.py (using your actual last name), and fill in the TODOs. Assume that your program will always use the dataset from the last assignment (a7a.*) with a single hidden layer.
2) Read through the command line interface below, which governs iterations, learning rate, development data, data locations, hidden layer size, and weight initialization. Make sure our implementation follows this interface.
3) Experiment: see how accuracy changes when you vary the hyperparameters, including number of iterations, learning rate, and hidden layer size. Use development data to pick the best set of hyperparameters.
4) Record what you did, why, and what the results were in your README. Discuss your interpretation of the results.
5) Download the smoke test script and run it in your submission directory. It will identify some common problems (but not necessarily all problems!). Remove it before submitting. NOTE: This script may not work on Windows.
6) With your code on the CSUG machine in directory submit_dir (name is not important), run this: /u/cs446/TURN_IN submit_dir


### Details:
-For the loss function, use log likelihood as shown in class.
-Use a batch size of 1 (i.e. update the weights after every data point).
-Use a network with only 1 hidden layer (of variable size), and a single-node output layer.
-Use sigmoid as the activation for both the hidden layer and output layer.
-Remember that there is a bias at every layer. The skeleton code will add the bias to the end of each input vector for you, and the weights files we provide via the --weights_files argument (see below) have bias weights in their final column.


### Interface:

    --iterations [int]
Stop training after this many iterations through the data (sometimes called epochs in the literature).

    --lr [float]
Learning rate.

    --nodev
When present, do NOT use development data. This means that your code should run for exactly the number of iterations specified in that argument (no early stopping). For this assignment, the autograder will always provide this argument, and will also always provide the --weights_files argument (see below). When this argument isn't provided, you should use the development data to control training.

    --weights_files [hidden_weights_filename] [output_weights_filename]
When present, initialize the two matrices with the values present in these files. The first is for the weight matrix between the input and the single hidden layer, and the second is for the weight matrix between the hidden layer and the output. Both files are in the format used by numpy.savetxt and numpy.loadtxt.

    --hidden_dim [int]
Only provided if --weights_files is NOT provided. Specifies the number of nodes in the network's single hidden layer. NOTE: the autograder will never use this. This argument is provided to you as a convenience.

    --print_weights
When present, print the two weight matrices. The skeleton file does this for you.

    --train_file [filename]
Load training data from here (default is /u/cs246/data/adult/a7a.train).

    --dev_file [filename]
Load dev data from here (default is /u/cs246/data/adult/a7a.dev).

    --test_file [filename]
Load test data from here (default is /u/cs246/data/adult/a7a.test).
