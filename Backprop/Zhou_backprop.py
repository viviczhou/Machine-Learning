#!/usr/bin/env python3
import numpy as np
from io import StringIO

NUM_FEATURES = 124 #features are 1 through 123 (123 only in test set), +1 for the bias
DATA_PATH = "/u/cs246/data/adult/" #TODO: if doing development somewhere other than the cycle server, change this to the directory where a7a.train, a7a.dev, and a7a.test are

#returns the label and feature value vector for one datapoint (represented as a line (string) from the data file)
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
    y = max(y,0) #treat -1 as 0 instead, because sigmoid's range is 0-1
    for t in tokens[1:]:
        parts = t.split(':')
        feature = int(parts[0])
        value = int(parts[1])
        x[feature-1] = value
    x[-1] = 1 #bias
    return y, x

#return labels and feature vectors for all datapoints in the given file
def parse_data(filename):
    with open(filename, 'r') as f:
        vals = [parse_line(line) for line in f]
        (ys, xs) = ([v[0] for v in vals],[v[1] for v in vals])
        return np.asarray([ys],dtype=np.float32).T, np.asarray(xs,dtype=np.float32).reshape(len(xs),NUM_FEATURES,1) #returns a tuple, first is an array of labels, second is an array of feature vectors

def init_model(args):
    w1 = None
    w2 = None

    if args.weights_files:
        with open(args.weights_files[0], 'r') as f1:
            w1 = np.loadtxt(f1)
        with open(args.weights_files[1], 'r') as f2:
            w2 = np.loadtxt(f2)
            w2 = w2.reshape(1,len(w2))
    else:
        #TODO (optional): If you want, you can experiment with a different random initialization. As-is, each weight is uniformly sampled from [-0.5,0.5).
        w1 = np.random.rand(args.hidden_dim, NUM_FEATURES) #bias included in NUM_FEATURES
        w2 = np.random.rand(1, args.hidden_dim + 1) #add bias column
        #w1 = np.random.uniform(low = -0.5, high = 0.5, size = (args.hidden_dim, NUM_FEATURES,))
        #w2 = np.random.uniform(low = -0.5, high = 0.5, size = (1, args.hidden_dim,))

    #At this point, w1 has shape (hidden_dim, NUM_FEATURES) and w2 has shape (1, hidden_dim + 1). In both, the last column is the bias weights.


    #TODO: Replace this with whatever you want to use to represent the network; you could use use a tuple of (w1,w2), make a class, etc.
    model = (w1,w2)
    #raise NotImplementedError #TODO: delete this once you implement this function
    return model

def train_model(model, train_ys, train_xs, dev_ys, dev_xs, args):
    #TODO: Implement training for the given model, respecting args
    old_train_accuracy = 0.0
    m = train_ys.shape[0]
    old_dev_accuracy = 0.0
    new_cost = 0.0
    if not args.nodev and not args.lr:
        print('Learning rate not inputed and using dev set...')
        etas = pow(10,np.linspace(-3, 0, num=10).reshape((10, )))
        acc_tracker = {}
        iter_tracker = {}
        w1_tracker = {}
        w2_tracker = {}
        for eta in etas:
            lr = eta
            print(lr)
            #if not args.iterations:
            #    iterations = 1000
           # if args.iterations:
            iterations = max(args.iterations, 1000)
            model = init_model(args)
            w1, w2 = extract_weights(model)
            for iteration in range(iterations):
                old_w1 = w1
                old_w2 = w2
                for x, y in zip(train_xs, train_ys):
                    y_hat, a_h, z_h, z_o = feedforward(w1, w2, x)
                    loss = log_likelihood(y, y_hat)
                    w1, w2 = backprop(x, y, y_hat, a_h, z_o, z_h, w1, w2, lr)
                    new_cost += loss
                model = (w1,w2)
                new_dev_accuracy = test_accuracy(model,dev_ys, dev_xs)
                new_train_accuracy = test_accuracy(model, train_ys, train_xs)
                if new_train_accuracy > old_train_accuracy and new_dev_accuracy < old_dev_accuracy:
                    acc_tracker[eta] = old_dev_accuracy
                    iter_tracker[eta] = iteration + 1
                    w1_tracker[eta] = old_w1
                    w2_tracker[eta] = old_w2
                    break
                else:
                    old_cost = new_cost
                    new_cost = 0.0
                    old_dev_accuracy = new_dev_accuracy
                    old_train_accuracy = new_train_accuracy
                if iteration + 1 == iterations:
                    acc_tracker[eta] = new_dev_accuracy
                    iter_tracker[eta] = iterations
                    w1_tracker[eta] = w1
                    w2_tracker[eta] = w2
                if iterations % 10 == 0:
                    print('Iteration #{}: {}.'.format(iteration,float(np.round(old_cost,2))))
        max_acc = max(acc_tracker.values())
        best_eta = [k for k,v in acc_tracker.items() if v == max_acc][0]
        best_iter = iter_tracker[best_eta]
        w1 = w1_tracker[best_eta]
        w2 = w2_tracker[best_eta]
        model = (w1, w2)
        print('The most suitable learning rate is:',best_eta,'\nThe best number of iterations for that learning rate is:',best_iter)
        return model
    elif args.lr and not args.nodev:
        print('Input learning rate and using dev set...')
        lr = args.lr
        iterations = args.iterations
        w1, w2 = extract_weights(model)
        for iteration in range(iterations):
            old_w1 = w1
            old_w2 = w2
            for x, y in zip(train_xs, train_ys):
                y_hat, a_h, z_h, z_o = feedforward(w1, w2, x)
                loss = log_likelihood(y, y_hat)
                w1, w2 = backprop(x, y, y_hat, a_h, z_o, z_h, w1, w2, lr)
                new_cost += loss
            model = (w1,w2)
            new_dev_accuracy = test_accuracy(model,dev_ys, dev_xs)
            new_train_accuracy = test_accuracy(model, train_ys, train_xs)
            if new_train_accuracy > old_train_accuracy and new_dev_accuracy < old_dev_accuracy:
                w1 = old_w1
                w2 = old_w2
                print('Iteration #{}: {}.'.format(iteration,float(np.round(old_cost,2))))
                best_iter = iteration
                print('The learning rate is:',lr,'\nThe best number of iterations for that learning rate is:',best_iter)
                model = (w1, w2)
                return model
            else:
                old_cost = new_cost
                new_cost = 0.0
                old_dev_accuracy = new_dev_accuracy
                old_train_accuracy = new_train_accuracy
            if iterations % 10 == 0:
                print('Iteration #{}: {}.'.format(iteration,float(np.round(old_cost,2))))
        print('Run',iterations,'iterations, and the model is still improving.\nCan try larger iterations...')
        model = (w1, w2)
        return model
    elif args.nodev:
        #print('Not using dev set...')
        w1, w2 = extract_weights(model)
        lr = args.lr
        iterations = args.iterations
        for iteration in range(iterations):
            for x, y in zip(train_xs, train_ys):
                y_hat, a_h, z_h, z_o = feedforward(w1, w2, x)
           #     loss = log_likelihood(y, y_hat)
                w1, w2 = backprop(x, y, y_hat, a_h, z_o, z_h, w1, w2, lr)
            #if iterations <= 10:
             #   print('Iteration #{}: {}.'.format(iteration,float(np.round(loss,2))))
            #else:
             #   if iteration % 10 == 0:
              #      print('Iteration #{}: {}.'.format(iteration,float(np.round(loss,2))))
        model = (w1, w2)
    #raise NotImplementedError #TODO: delete this once you implement this function
    return model

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    return (1-sigmoid(z))*sigmoid(z)

def log_likelihood(y, y_hat):
    z = np.dot(y,y_hat)
    p = sigmoid(z)
    loss = -np.log(p)
    return loss

# def log_derivative():

def feedforward(w1, w2, x):
    z_h = np.dot(w1,x)
    a_h = np.append(sigmoid(z_h),1)
    z_o = np.dot(w2,a_h)
    y_hat = sigmoid(z_o)
    return y_hat, a_h, z_h, z_o

def backprop(x, y, y_hat, a_h, z_o, z_h, w1, w2, lr):
    #delta1 = (1/sigmoid(y * z_o)) * sigmoid_derivative(y * z_o)*y
    #delta1 = (1/y_hat)*sigmoid_derivative(z_o)
    dy = ((-y)/y_hat) + ((1-y)/(1-y_hat))
    delta2 = np.dot(dy,sigmoid_derivative(z_o))
    dw2 = delta2 * a_h
    delta1 = (np.transpose(w2)[:-1] * delta2) * sigmoid_derivative(z_h)
    dw1 = np.dot(delta1,x.T)
    w1 = w1 - lr * dw1
    w2 = w2 - lr * dw2
    return w1, w2

def test_accuracy(model, test_ys, test_xs):
    accuracy = 0.0
    #TODO: Implement accuracy computation of given model on the test data
    w1, w2 = extract_weights(model)
    y_pred = []
    for x in test_xs:
        y_pred.append(feedforward(w1,w2,x)[0])
    m = test_ys.shape[0]
    for i in range(m):
        if y_pred[i] <= 0.5:
            y_pred[i] = 0
        else:
            y_pred[i] = 1
    y_pred = np.transpose(np.array([y_pred]))
    accuracy = float(sum(y_pred == test_ys)/m)
    #raise NotImplementedError #TODO: delete this once you implement this function
    return accuracy

def extract_weights(model):
    w1 = None
    w2 = None
    #TODO: Extract the two weight matrices from the model and return them (they should be the same type and shape as they were in init_model, but now they have been updated during training)
    (w1, w2) = model
    #raise NotImplementedError #TODO: delete this once you implement this function
    return w1, w2

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Neural network with one hidden layer, trainable with backpropagation.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, #default=0.1, 
            help='Learning rate to use for update in training loop.')

    weights_group = parser.add_mutually_exclusive_group()
    weights_group.add_argument('--weights_files', nargs=2, metavar=('W1','W2'), type=str, help='Files to read weights from (in format produced by numpy.savetxt). First is weights from input to hidden layer, second is from hidden to output.')
    weights_group.add_argument('--hidden_dim', type=int, default=5, help='Dimension of hidden layer.')

    parser.add_argument('--print_weights', action='store_true', default=False, help='If provided, print final learned weights to stdout (used in autograding)')

    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH,'a7a.train'), help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH,'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH,'a7a.test'), help='Test data file.')


    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.weights_files: iterable of str; if present, contains two fields, the first is the file to read the first layer's weights from, second is for the second weight matrix.
    args.hidden_dim: int; number of hidden layer units. If weights_files is provided, this argument should be ignored.
    args.train_file: str; file to load training data from.
    args.dev_file: str; file to load dev data from.
    args.test_file: str; file to load test data from.
    """
    train_ys, train_xs = parse_data(args.train_file)
    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs= parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)

    model = init_model(args)
    model = train_model(model, train_ys, train_xs, dev_ys, dev_xs, args)
    accuracy = test_accuracy(model, test_ys, test_xs)
    print('Test accuracy: {}'.format(accuracy))
    if args.print_weights:
        w1, w2 = extract_weights(model)
        with StringIO() as weights_string_1:
            np.savetxt(weights_string_1,w1)
            print('Hidden layer weights: {}'.format(weights_string_1.getvalue()))
        with StringIO() as weights_string_2:
            np.savetxt(weights_string_2,w2)
            print('Output layer weights: {}'.format(weights_string_2.getvalue()))

if __name__ == '__main__':
    main()

