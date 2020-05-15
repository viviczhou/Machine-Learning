# Perceptron Algorithm

## 1. Introduction: 

Predict whether income exceeds $50K/yr based on census data. Also known as "Census Income" dataset.

First column is class label, remaining columns are a sparse representation
of the feature vector in format <feature>:<value>.  All other features are 0.

more information on the task:
http://archive.ics.uci.edu/ml/datasets/Adult

preprocessed version from:
http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html



## 2. Files Included:

Zhou_perceptron.py: The python file filled all the TODOs.

train_dev_acc_iter.png: A plot for the accuracies of training set and development set. Shows how accuracy changes with the number of iterations (from 1 iterations to 110 iterations) when learning rate is constant (equals to 2).

train_dev_acc_lr.png: A plot for the accuracies of training set and development set. Shows how accuracy changes with the learning rate (from 0.0 to 2.0 with step = 0.02),when number of iterations is constant (equals to 25).

test_acc_iter.png: A plot for the accuracies of testing set. Shows how accuracy changes with the number of iterations (from 1 iterations to 110 iterations) when learning rate is constant (equals to 2).

test_acc_lr.png: A plot for the accuracies of testing set. Shows how accuracy changes with the learning rate (from 0.0 to 2.0 with step = 0.02) when number of iterations is constant (equals to 25).

README.md: This file.



## 3. Experiments & Discussions:

### 3.1 How performance changes as a function of number of iterations:

With the constant learning rate (lr = 2), the model converged when number of iterations is approximately 2. Before convergence, the prediction accuracy of the model increases when the number of iterations increases. After convergence, the prediction accuracy float around 80%. I found the accuracy does not change with the number of iterations after convergence.

The trends for all three datasets (training, development, and testing) are same.

The results are visualized. Please see train_dev_acc_iter_uploaded.png and test_acc_iter_uploaded.png for more details.


### 3.2 How performance changes as a function of learning rates:

With the constant number of iterations (iterations = 25), the accuracy of the model floats around 80% and does not change with the learning rate.

The trends for all three datasets (training, development, and testing) are same.

The results are visualized. Please see train_dev_acc_lr_uploaded.png and test_acc_lr_uploaded.png for more details.

(Notes: the learning rate is range from 0.0 to 0.02 with step size equals to 0.02 for experiments on all three datasets. The x axis is labeled from 0 to 100 in the plot. The learning rate is calculated as lr = 0.02*xtick)


### 3.3 How to best use dev data during training:

When run command without '--nodev', the dev data will be used to avoid overfitting. In theory, the model accuracy should increase with the number of iterations. When accuracy of the dev data decreases with the number of iterations, the training process should be early stoped. Considered that there will be fluctuations in model accuracy when the number of iterations changes, so using accuracy alone may not be a signal that sufficient enough. Thus, I use the loss and accuracy of dev data together as the signal of overfitting.

When loss decreases and accuracy also decreases with the number of iterations, the training process should be early stopped.



## 4. Additional Information:

When run command line: './Zhou_perceptron.py', the model will return some outputs using a suitable learning rate I found and will stop before the model is overfitted.

The output is listed below:
	
	The most suitable learning rate is: 0.6579332246575682 
	The best number of iterations for that learning rate is: 2
	Test accuracy: 0.8231887483748966
	Feature weights (bias last): -3.3367297036392713 -2.502295659009148 1.0567497708243914 2.323551502912072 -0.10919202840790998 0.1866709439876717 0.30424397734371134 2.022233760876124 4.380927565513076 1.6358461500891364 0.5417762921944258 -3.832405165108013 0.0 -4.489047927633336 2.2922766925119458 0.9459254371981891 -1.2043256264110211 -0.11274469298564416 -1.9054698586948609 0.39870749182298404 1.037233653060118 -0.46024155153244295 1.1548717612007482 -0.17351936273197333 0.6434635794888071 -0.16620040444944006 -2.529376795607087 4.31366466033629 -0.4718505620239253 2.2299142696240573 -0.8799276314988378 3.911747293289733 -0.49471455944772535 -9.176218100156314 -5.6656249081389385 -0.46024155153244295 0.39870749182298404 0.46994421675683395 2.6892986337716964 3.038229265148038 -2.133184043726912 -4.890153401382071 -0.5699536267951404 -0.28249859233558294 -3.0924436696777775 5.36208795144958 2.6131307731061497 -0.3375668593803536 -1.024257978383813 1.4840198362898096 6.200881297240535 -0.23552410949242297 -1.0770019804039461 0.7817254738786634 -1.1270455150040875 -2.062661244607158 -1.1462057475317002 -3.4458291695918932 4.615628748776353 0.0 2.8358483813369553 -0.6340465680834735 -1.4381113784850947 -0.6414956271630168 -2.4251369006841754 -0.2649740242410621 0.48236479134989296 1.7220788138526462 -0.10975288798514637 -1.6737661644382535 -2.9888406700990044 -2.8862822970220012 0.3183661797021371 -3.784147889303582 1.2162317719837143 -3.3272945708482187 0.7593784535283539 -3.713765698438527 0.06265287457539359 -1.3243735072335086 0.11897161854890226 2.2885985952278753 3.643212614213451 5.903755261967348 1.1130438838943308 -3.8800309605839347 4.172434797677788 -1.1316959567141531 0.0 -6.951367860331413 1.3326672196734197 -0.8675749028181039 -1.6228201464582506 -0.7626776349958795 3.4552023650759063 -2.0296538893499214 0.0 2.408062277494344 3.6204176286543372 0.836660170740763 0.9710037133477023 -4.658715388191803 0.31490286723630345 -1.858482039191302 5.565807965434864 0.25666329650457753 -1.9839953660585916 -3.5090300095169966 6.724487637535352 -1.8355404637265995 2.4524363596458265 -4.78657375151627 -2.5879898312181675 3.5067221724906785 4.189356742752337 4.749443520104681 -3.926710773476377 6.527191145142389 0.9642458533490951 -2.775367198013533 -6.567519646310011 -1.5633700819447942 0.0 -2.5679161173198652 
	

When run command line: 

	'./Zhou_perceptron.py --nodev'
the model will return some outputs using the default learning rate and default iterations. The model will not early stop.

The output is listed below:

	Test accuracy: 0.8136154118898475
	Feature weights (bias last): -6.0 -6.0 5.0 3.0 0.0 1.0 0.0 5.0 9.0 2.0 0.0 -8.0 0.0 -9.0 6.0 0.0 -2.0 1.0 -4.0 2.0 3.0 -1.0 1.0 -1.0 2.0 0.0 -3.0 7.0 1.0 3.0 3.0 7.0 -2.0 -22.0 -11.0 -1.0 2.0 1.0 5.0 6.0 -3.0 -8.0 -3.0 -1.0 -3.0 8.0 3.0 1.0 -1.0 3.0 10.0 -1.0 0.0 1.0 0.0 -5.0 -4.0 -5.0 7.0 0.0 5.0 0.0 -3.0 -1.0 -5.0 0.0 1.0 4.0 -3.0 -4.0 -2.0 -4.0 0.0 -8.0 4.0 -6.0 2.0 -7.0 0.0 -3.0 2.0 4.0 5.0 12.0 4.0 -7.0 7.0 -1.0 0.0 -9.0 6.0 -1.0 -1.0 -3.0 7.0 -2.0 0.0 3.0 5.0 3.0 3.0 -8.0 1.0 -6.0 7.0 0.0 -6.0 -8.0 12.0 0.0 2.0 -9.0 -6.0 4.0 7.0 7.0 -7.0 10.0 1.0 -4.0 -11.0 -2.0 0.0 -4.0

When run command with 

	'--nodev --iterations # --lr #'
the model will return some outputs using the given learning rate and given iterations. The model will not early stop.

When run command 

	'--iterations # --lr #' without '--nodev'
the model will return some outputs using the given learning rate and given iterations. The model will early stop.
