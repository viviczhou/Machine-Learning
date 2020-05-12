#!/usr/bin/python3
import numpy as np
#TODO: understand that you should not need any other imports other than those already in this file; if you import something that is not installed by default on the csug machines, your code will crash and you will lose points

NUM_FEATURES = 124 #features are 1 through 123 (123 only in test set), +1 for the bias
DATA_PATH = "/u/cs246/data/adult/" #TODO: if you are working somewhere other than the csug server, change this to the directory where a7a.train, a7a.dev, and a7a.test are on your machine

#returns the label and feature value vector for one datapoint (represented as a line (string) from the data file)
def parse_line(line):
    tokens = line.split()
    x = np.zeros(NUM_FEATURES)
    y = int(tokens[0])
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
        return np.asarray(ys), np.asarray(xs) #returns a tuple, first is an array of labels, second is an array of feature vectors

def perceptron(train_ys, train_xs, dev_ys, dev_xs, args):
    weights = np.zeros(NUM_FEATURES)
    #TODO: implement perceptron algorithm here, respecting args
    old_cost = float('inf')
    old_accuracy = 0.0
    m = train_ys.shape[0]
    if args.nodev:
        lr = args.lr
        iterations = args.iterations
    if not args.nodev:
        etas = pow(10,np.linspace(-3, 0, num=100).reshape((100, )))
        acc_tracker = {}
        iter_tracker = {}
        weights_tracker = {}
        for eta in etas:
            lr = eta
            iterations = 1000
            for iteration in range(iterations):
                old_weights = weights
                for i in range(m):
                    if (np.dot(train_xs[i],weights)*train_ys[i])<=0:
                        weights = weights + lr*train_xs[i]*train_ys[i]
                y_hat = np.dot(dev_xs,weights)
                loss = y_hat-dev_ys
                new_cost = np.dot(loss.T,loss) + np.dot(weights.T,weights)
                new_accuracy = test_accuracy(weights,dev_ys,dev_xs)
                if new_cost < old_cost and new_accuracy < old_accuracy:
                    acc_tracker[eta] = old_accuracy
                    iter_tracker[eta] = iteration+1
                    weights_tracker[eta] = old_weights
                    break
                else:
                    old_cost = new_cost
                    old_accuracy = new_accuracy
                if iteration+1 == iterations:
                    acc_tracker[eta] = old_accuracy
                    iter_tracker[eta] = iteration+1
                    weights_tracker[eta] = weights
        max_acc = max(acc_tracker.values())
        best_eta = [k for k,v in acc_tracker.items() if v == max_acc][0]
        best_iter = iter_tracker[best_eta]
        weights = weights_tracker[best_eta]
        print('The most suitable learning rate is:',best_eta,'\nThe best number of iterations for that learning rate is:',best_iter)
        return weights
    for iteration in range(iterations):
        old_weights = weights
        for i in range(m):
            if (np.dot(train_xs[i],weights)*train_ys[i])<=0:
                weights = weights + lr*train_xs[i]*train_ys[i]
    return weights

def test_accuracy(weights, test_ys, test_xs):
    accuracy = 0.0
    #TODO: implement accuracy computation of given weight vector on the test data (i.e. how many test data points are classified correctly by the weight vector)
    m = test_ys.shape[0]
    for i in range(m):
        if (np.dot(test_xs[i],weights)*test_ys[i])>0:
            accuracy += 1
    accuracy = accuracy/m
    return accuracy

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Basic perceptron algorithm.')
    parser.add_argument('--nodev', action='store_true', default=False, help='If provided, no dev data will be used.')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations through the full training data to perform.')
    parser.add_argument('--lr', type=float, default=1.0, help='Learning rate to use for update in training loop.')
    parser.add_argument('--train_file', type=str, default=os.path.join(DATA_PATH,'a7a.train'), help='Training data file.')
    parser.add_argument('--dev_file', type=str, default=os.path.join(DATA_PATH,'a7a.dev'), help='Dev data file.')
    parser.add_argument('--test_file', type=str, default=os.path.join(DATA_PATH,'a7a.test'), help='Test data file.')
    args = parser.parse_args()

    """
    At this point, args has the following fields:

    args.nodev: boolean; if True, you should not use dev data; if False, you can (and should) use dev data.
    args.iterations: int; number of iterations through the training data.
    args.lr: float; learning rate to use for training update.
    args.train_file: str; file name for training data.
    args.dev_file: str; file name for development data.
    args.test_file: str; file name for test data.
    """
    train_ys, train_xs = parse_data(args.train_file)
    dev_ys = None
    dev_xs = None
    if not args.nodev:
        dev_ys, dev_xs= parse_data(args.dev_file)
    test_ys, test_xs = parse_data(args.test_file)
    weights = perceptron(train_ys, train_xs, dev_ys, dev_xs, args)
    accuracy = test_accuracy(weights, test_ys, test_xs)
    print('Test accuracy: {}'.format(accuracy))
    print('Feature weights (bias last): {}'.format(' '.join(map(str,weights))))

if __name__ == '__main__':
    main()
