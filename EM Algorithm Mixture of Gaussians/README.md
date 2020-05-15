# EM Algorithm of Mixture of Gaussians


## 1. Introduction:

Gaol: Maximize log likelihood
Algorithm: EM Algorithm

In this assignment, I conducted experiments with different values of --cluster_num and --iterations, implemented tied covariances as well as the standard, separate covariance setting, and took note of trends in both training and development data. 



## 2. Files Included:


Zhou_em_gaussian.py: The python file filled all the TODOs.

LL_cluster_num.png: A plot for the log likelihood of both training and dev sets. Shows how log likelihood changes with the number of clusters (from 1 to 10 with step = 1) when number of iterations (equals to 100) is constant.

LL_iteration.png: A plot for the log likelihood of both training and dev sets. Shows how log likelihood changes with the number of iterations (from 1 to 100 with step = 1).

README: This file



## 3. Experiments & Discussions:
### 3.1 How log likelihood varies with number of iterations:

#### Why: 
Try to find the trend of the log likelihood with the number of iterations.

#### Results and interpretation:
The model converged when the number of iterations is approximately 20. Before convergence, the log likelihood of the model increases with the number of iterations. After convergence, the log likelihood of training set is around -4.55, and that of dev set is around -4.65. I found the log likelihood does not change with the number of iterations after convergence. The model perform best with iterations equals to 20, when it provides both the largest log likelihood and utilizes the least running time.

When using tied covariances, the model converges faster with lower maximum log likelihood. I used cluster number s equal to 3, 5, and 10 respectively. It converged when the number of iterations is approximately 10 regardless of the number of clusters. Both log likelihood of training set and dev set is around -10.60. I found the log likelihood does not change with the number of iterations after convergence. The model perform best with iterations equals to 10, when it provides both the largest log likelihood and utilizes the least running time.


### 3.2 How log likelihood varies with number of clusters:

#### Why: 
Try to find the trend of the log likelihood with the number of clusters.

#### Results and interpretation:
With the constant number of iterations (iterations = 100), the model performed best with the number of clusters (K) equal to 10. When K < 4, the log likelihood increases with K. When K >= 4, the log likelihood float around -4.35. When K = 10, both training set and dev set have large log likelihood, and the value is about -4.35 for both sets.

When using tied covariances, the performance of model does not change with the number of clusters. The log likelihood of training set is always mourned 0.00003, and that of dev set is always around -0.00001. Both are larger than the value of their counter data set when standard covariances are used.

The results are visualized. Please see LL_cluster_num.png and LL_cluster_num_tied.png for more details.
