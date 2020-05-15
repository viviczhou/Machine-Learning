# EM Algorithm of Mixture of Gaussians


## 1. Introduction:

Gaol: Maximize log likelihood
Algorithm: EM Algorithm

In this assignment, I conducted experiments with different values of --cluster_num and --iterations, implemented tied covariances as well as the standard, separate covariance setting, and took note of trends in both training and development data. 



## 2. Files Included:


Zhou_em_gaussian.py: The python file filled all the TODOs.

LL_cluster_num.png: A plot for the log likelihood of both training and dev sets. Shows how log likelihood changes with the number of clusters (from 1 to 10 with step = 1) when number of iterations (equals to 100) is constant.

LL_cluster_num_tied.png: A plot for the log likelihood of both training and dev sets. Shows how log likelihood changes with the number of clusters (from 1 to 10 with step = 1) when number of iterations (equals to 100) is constant and the covariance setting is tied.

LL_iteration.png: A plot for the log likelihood of both training and dev sets. Shows how log likelihood changes with the number of iterations (from 1 to 100 with step = 1).

tiedk10_ll_iter.png: A plot for the log likelihood of both training and dev sets. Shows how log likelihood changes with the number of iterations (from 1 to 100 with step = 1) when number of clusters (equals to 10) is constant and the covariance setting is tied.

tiedk5_ll_iter.png: A plot for the log likelihood of both training and dev sets. Shows how log likelihood changes with the number of iterations (from 1 to 100 with step = 1) when number of clusters (equals to 5) is constant and the covariance setting is tied.

tiedk3_ll_iter.png: A plot for the log likelihood of both training and dev sets. Shows how log likelihood changes with the number of iterations (from 1 to 100 with step = 1) when number of clusters (equals to 3) is constant and the covariance setting is tied.

README: This file

points.dat: Provided by the Instructors
