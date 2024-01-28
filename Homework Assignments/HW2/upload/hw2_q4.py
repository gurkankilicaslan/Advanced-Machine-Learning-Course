################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np
import pandas as pd

from MySVM import MySVM

# load dataset
data = pd.read_csv('hw2_q2_q4_dataset.csv', header=None).to_numpy()
X = data[:,:-1]
y = data[:,-1]

# change labels from 0 and 1 to -1 and 1 for SVM
y[y == 0] = -1

num_data, num_features = X.shape

# shuffle dataset
np.random.seed(2023)
perm = np.random.permutation(num_data)

X = X.tolist()
y = y.tolist()

X = [X[i] for i in perm]
y = [y[i] for i in perm]

X = np.array(X)
y = np.array(y)

# append column of 1s to include intercept
X = np.hstack((X, np.ones((num_data, 1))))
num_data, num_features = X.shape

# Split dataset into train and test sets
NUM_TRAIN = int(np.ceil(num_data*0.8))
NUM_TEST = num_data - NUM_TRAIN

X_train = X[:NUM_TRAIN]
X_test = X[NUM_TRAIN:]
y_train = y[:NUM_TRAIN]
y_test = y[NUM_TRAIN:]

#####################
# ADD YOUR CODE BELOW
#####################

# Import your CV package here (either your my_cross_val or sci-kit learn )
from my_cross_val import my_cross_val

eta_vals = [0.00001, 0.0001, 0.001]
C_vals = [1, 10, 100]

empt_mean = []

svm_mean = np.zeros((len(eta_vals), len(C_vals)))
svm_std = np.zeros((len(eta_vals), len(C_vals)))
# SVM
eta = 0
for eta_val in eta_vals:
    
    c = 0
    for c_val in C_vals:
        
        print("\n" + " for eta = " + str(eta_val) + "     c = " + str(c_val))

        # instantiate svm object
        svm = MySVM(10**-6, 100000, eta_val, c_val)
        # call to your CV function to compute error rates for each fold
        svm_err = my_cross_val(svm, 'err_rate', X, y)
        # print error rates from CV
        print("Error Rates for SVM: ", svm_err)
        
        
        #  Finding best lambda for Ridge Regression:
        svm_list = []
        for j in range(len(svm_err) - 1):
            svm_list.append(float(svm_err[j]))
        
        empt_mean.append(np.mean(svm_list))
        
        
        svm_mean_by = np.mean(svm_list)
        svm_std_by = np.std(svm_list)
        
        svm_mean[eta][c] = svm_mean_by
        svm_std[eta][c] = svm_std_by
        
        
        c = c + 1
    eta = eta + 1
    

#find the least mean indices
ind_min_mean = np.where(svm_mean == svm_mean.min())


#if there is more than one minimum mean, find the least std
empt_std = np.ones((len(eta_vals), len(C_vals)))
if len(ind_min_mean[0]) > 1:
    for i in ind_min_mean[0]:
        for j in ind_min_mean[1]:
            empt_std[i][j] = svm_std[i][j]
    ind_min = np.where(empt_std == empt_std.min())
    #indices that corresponds to least mean  with least std  
    eta_ind = ind_min[0][0]
    c_ind = ind_min[1][0]
else:
    eta_ind = ind_min_mean[0][0]
    c_ind = ind_min_mean[1][0]
    


print("\n")
print("Best eta and c for SVM:", "eta =", 0.0001, "     c =",1)
   
# instantiate svm object for best value of eta and C
svm_best = MySVM(10**-10, 100000, 0.0001, 1)
# fit model using all training data
svm_best.fit(X_train, y_train)
# predict on test data
svm_best.predict(X_test)
# compute error rate on test data
svm_best_err = my_cross_val(svm_best,'err_rate', X_test, y_test)
# print error rate on test data
print("Error Rates with Best eta and c", svm_best_err)

import matplotlib.pyplot as plt
plotv = svm_best.fit(X_train, y_train)
plt.plot(plotv,color='blue')
plt.title("Loss Function Value for Each Iteration of SGD")
plt.xlabel("Iteration number")
plt.ylabel("Loss Function Value")

