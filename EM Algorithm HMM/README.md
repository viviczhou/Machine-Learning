# EM Algorithm for HMM of Mixture of Gaussians

## 1. Introduction:

Gaol: Maximize log likelihood
Algorithm: EM Algorithm (HMM)

In this assignment, I conducted experiments with different values of --cluster_num and --iterations, implemented tied covariances as well as the standard, separate covariance setting, and took note of trends in both training and development data, compared HMM model and the original non-sequence model to see which one performs better, and found the best number of states.


## 2. Files Included:

Zhou_hmm_gaussian.py: The python file filled all the TODOs.

LL_K.png: A plot for the log likelihood of both training and dev sets. Shows how log likelihood changes with the number of clusters (from 2 to 11 with step = 1) when number of iterations (equals to 50) is constant.

LL_iter.png: A plot for the log likelihood of both training and dev sets. Shows how log likelihood changes with the number of iterations (from 1 to 100 with step = 1).

model_compare.png: A plot for the log likelihood of two models (HMM and original non-sequence model). Shows how log likelihood changes with iterations (from 1 to 100 with step = 1) in both models.

README: This file


## 3. Experiments & Discussions:
### 3.1 How log likelihood varies with number of iterations:

#### Why: 
Try to find the trend of the log likelihood with the number of iterations.

#### Results and interpretation:
The model converged when the number of iterations is approximately 40. Before convergence, the log likelihood of the model increases with the number of iterations. After convergence, the log likelihood of training set is around -4.35, and that of dev set is around -4.45. I found the log likelihood does not change with the number of iterations after convergence. The model perform best with iterations equals to 40, when it provides both the largest log likelihood and utilizes the least running time.

The best number of states is 5.

When using tied covariances, the model converges faster with lower maximum log likelihood. I used cluster number s equal to 2, 7, and 10 respectively. It converged when the number of iterations is approximately 10 regardless of the number of clusters. Both log likelihood of training set and dev set is around -4.80. I found the log likelihood does not change with the number of iterations after convergence. The model perform best with iterations equals to 10, when it provides both the largest log likelihood and utilizes the least running time.

The results are visualized. Please see LL_iter.png for more details.


### 3.2 How log likelihood varies with number of clusters:

#### Why: 
Try to find the trend of the log likelihood with the number of clusters.

#### Results and interpretation:
With the constant number of iterations (iterations = 50), the model performed best with the number of clusters (K) equal to 5. When K < 5, the log likelihood increases with K. When K >= 5, the log likelihood of training set increases with K while that of dev set decreases with the K reflecting the fact of overfitting. When K = 5, both the dev set and the training set have log likelihood = -3.70.  

When using tied covariances, the performance of model does not change with the number of clusters. The log likelihood of training set is always mourned -4.78, and that of dev set is always around -4.83. Both are larger than the value of their counter data set when standard covariances are used.

The results are visualized. Please see LL_cluster_num.png for more details.


### 3.3 Does the HMM model the data better than the original non-sequence model?

The log likelihood of HMM model is higher than that of the original model (both training set and dev set) when K is same and number of iterations is also the same. Thus, we can conclude that HMM model is better than the original non-sequence model. 

The results are visualized. Please see model_compare.png for more details.
