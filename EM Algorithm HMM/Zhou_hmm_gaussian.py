#!/usr/bin/env python3
import numpy as np
if not __file__.endswith('_hmm_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as Lastname_hmm_gaussian.py (replacing Lastname with your last name)!')
    exit(1)

DATA_PATH = "/u/cs446/data/em/"

def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9*len(data))
    train_xs = np.asarray(data[:dev_cutoff],dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:],dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs

def init_model(args):
    if args.cluster_num:
        mus = np.zeros((args.cluster_num,2))
        if not args.tied:
            sigmas = np.zeros((args.cluster_num,2,2))
        else:
            sigmas = np.zeros((2,2))
        transitions = np.zeros((args.cluster_num,args.cluster_num)) #transitions[i][j] = probability of moving from cluster i to cluster j
        initials = np.zeros(args.cluster_num) #probability for starting in each state
        
        # Set Seed
        seed = 1
        np.random.seed(seed)
        # Initialize mus
        mus = np.random.uniform(0.0,1.0,(mus.shape))
        # Initialize sigmas
        if not args.tied:
            for k in range(args.cluster_num):
                sigmas[k] = np.eye(2)
        else:
            sigmas = np.eye(2)
        # Initialize initials
        initials = np.tile(1/args.cluster_num,(args.cluster_num))
        # Initialize transitions
        transitions = np.random.rand(args.cluster_num, args.cluster_num)
        transitions = transitions/transitions.sum(axis=1, keepdims= 1) # Normalize each row to sum up to 1
    else:
        mus = []
        sigmas = []
        transitions = []
        initials = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #initial mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1 transition_this_to_0 transition_this_to_1 ... transition_this_to_K-1
                vals = list(map(float,line.split()))
                initials.append(vals[0])
                mus.append(vals[1:3])
                sigmas.append([vals[3:5],vals[5:7]])
                transitions.append(vals[7:])
        initials = np.asarray(initials)
        transitions = np.asarray(transitions)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(initials)

    model = {'mus': mus, 'sigmas': sigmas, 'initials': initials, 'transitions': transitions}
    return model

def forward(model, data, args):
    from scipy.stats import multivariate_normal
    from math import log
    alphas = np.zeros((len(data),args.cluster_num))
    log_likelihood = 0.0
    # Calculate and return forward probabilities (normalized at each timestep; see next line) and log_likelihood
    # This function is used to calculate alphas
    # To avoid numerical problems, calculate the sum of alpha[t] at each step, normalize alpha[t] by that value, and increment log_likelihood by the log of the value you normalized by.
    # This will prevent the probabilities from going to 0, and the scaling will be cancelled out in train_model when normalize
    initials, transitions, mus, sigmas = extract_parameters(model)
    T = len(data)
    for t in range(T):
        if t == 0:
            for i in range(args.cluster_num):
                if not args.tied:
                    alphas[t, i] = initials[i] * multivariate_normal.pdf(x = data[t], mean = mus[i], cov = sigmas[i])
                else:
                    alphas[t, i] = initials[i] * multivariate_normal.pdf(x = data[t], mean = mus[i], cov = sigmas)
        else:
            for i in range(args.cluster_num):
                if not args.tied:
                    alphas[t, i] = np.sum(alphas[t-1,:] * transitions[:,i]) * multivariate_normal.pdf(x = data[t], mean = mus[i], cov = sigmas[i])
                else:
                    alphas[t, i] = np.sum(alphas[t-1,:] * transitions[:,i]) * multivariate_normal.pdf(x = data[t], mean = mus[i], cov = sigmas)
        # Normalize alphas
        Z = np.sum(alphas[t,:])
        alphas[t,:] = alphas[t,:]/np.sum(alphas[t,:])
        # Increment log_likelihood
        log_likelihood = log_likelihood + log(Z)
    return alphas, log_likelihood

def backward(model, data, args):
    from scipy.stats import multivariate_normal
    betas = np.zeros((len(data),args.cluster_num))
    # Calculate and return backward probabilities (normalized like in forward before)
    # This function is used to calculate betas
    initials, transitions, mus, sigmas = extract_parameters(model)
    T = len(data)-1
    betas[T,:] = 1
    for t in reversed(range(T)):
        for i in range(args.cluster_num):
            beta = 0
            for j in range(args.cluster_num):
                if not args.tied:
                    beta += betas[t+1,j] * transitions[i,j] * multivariate_normal.pdf(x = data[t+1], mean = mus[j], cov = sigmas[j])
                else:
                    beta += betas[t+1,j] * transitions[i,j] * multivariate_normal.pdf(x = data[t+1], mean = mus[j], cov = sigmas)
            betas[t,i] = beta
        # Normalize betas
        betas[t,:] = betas[t,:]/np.sum(betas[t,:])
    return betas

def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    # train the model, respecting args (note that dev_xs is None if args.nodev is True)
    T = train_xs.shape[0]
    K = args.cluster_num
    initials, transitions, mus, sigmas = extract_parameters(model)
    for iter in range(args.iterations):
        gammas = np.zeros((T, K))
        ksi_matrix = np.zeros((T, K, K))
        alphas,_ = forward(model, train_xs, args)
        betas = backward(model, train_xs, args)
        # E-step
        for t in range(T):
            for i in range(K):
                for j in range(K):
                    if t != 0:
                        if not args.tied:
                            ksi_matrix[t,i,j] = alphas[t-1,i] * betas[t,j] * transitions[i,j] * multivariate_normal.pdf(x = train_xs[t], mean = mus[j], cov = sigmas[j])
                        else:
                            ksi_matrix[t, i, j] = alphas[t-1, i] * betas[t, j] * transitions[i, j] * multivariate_normal.pdf(x=train_xs[t], mean=mus[j], cov=sigmas)
                gammas[t, i] = alphas[t, i] * betas[t, i]
            # Normalize gammas
            gammas[t,] = gammas[t,] / np.sum(gammas[t,])
            # Normalize ksi
            if t != 0:
                ksi_matrix[t,] = ksi_matrix[t,]/np.sum(ksi_matrix[t,])
        # M-step
        for i in range(K):
            initials[i] = gammas[0,i]
            Z = np.sum(gammas[:,i])
            mus[i] = np.dot(gammas[:,i], train_xs)/Z
            if not args.tied:
                sigmas[i] = np.dot(gammas[:,i] * (train_xs - mus[i]).T, (train_xs - mus[i]))/Z
            else:
                sigmas += np.dot(gammas[:,i] * (train_xs - mus[i]).T, (train_xs - mus[i]))
            for j in range(K):
                transitions[i,j] = np.sum(ksi_matrix[:,i,j])/np.sum(ksi_matrix[:,i,:])

        if args.tied:
            sigmas = sigmas/T
    model = {'mus': mus, 'sigmas': sigmas, 'initials': initials, 'transitions': transitions}
    return model

def average_log_likelihood(model, data, args):
    # implement average LL calculation (log likelihood of the data, divided by the length of the data)
    ll = 0.0
    _,log_likelihood = forward(model, data, args)
    ll = log_likelihood/len(data)
    return ll

def extract_parameters(model):
    # Extract initials, transitions, mus, and sigmas from the model and return them (same type and shape as in init_model)
    initials = model['initials']
    transitions = model['transitions']
    mus = model['mus']
    sigmas = model['sigmas']
    return initials, transitions, mus, sigmas

def main():
    import argparse
    import os
    print('Gaussian') 
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true', help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied',action='store_true',help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print('You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    train_xs, dev_xs = parse_data(args)
    model = init_model(args)
    model = train_model(model, train_xs, dev_xs, args)
    nll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(nll_train))
    if not args.nodev:
        nll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(nll_dev))
    initials, transitions, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Initials: {}'.format(intersperse(' | ')(np.nditer(initials))))
        print('Transitions: {}'.format(intersperse(' | ')(map(intersperse(' '),transitions))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
    main()
