################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np
import pandas as pd

from MyLogisticRegression import MyLogisticRegression

# load dataset
data = pd.read_csv('hw2_q2_q4_dataset.csv', header=None).to_numpy()
X = data[:,:-1]
y = data[:,-1]

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
eta_vals = [ 0.00001, 0.0001, 0.001, 0.01, 0.1]


#suppress warnings




log_mean = []
log_std = []
# Logistic Regression
for eta_val in eta_vals:
    print("\n" + " for eta = " + str(eta_val) + " :")
    # instantiate logistic regression object
    log_reg = MyLogisticRegression(X.shape[1], 100000, eta_val)
    # call to CV function to compute error rates for each fold
    log_reg_err = my_cross_val(log_reg, 'err_rate', X, y)
    # print error rates from CV
    print("Error Rates for Logistic Regression: ", log_reg_err)
    
    
    #  Finding best lambda for Ridge Regression:
    log_list = []
    for j in range(len(log_reg_err) - 1):
        log_list.append(float(log_reg_err[j]))
    log_mean_by_eta = np.mean(log_list)
    log_std_by_eta = np.std(log_list)
    
    log_mean.append(log_mean_by_eta)
    log_std.append(log_std_by_eta)
    idx_min_log_mean = log_mean.index(min(log_mean))
    best_log_eta = eta_vals[idx_min_log_mean]
    
print("\n")
print("Best eta for Logistic Regression is ", best_log_eta)
    
# instantiate logistic regression object for best value of eta
log_reg_best = MyLogisticRegression(d = 10**-6, max_iters = 100000, eta_val = best_log_eta)
# fit model using all training data
log_reg_best.fit(X_train, y_train)
# predict on test data
log_reg_best.predict(X_test)
# compute error rate on test data
best_log_reg_err = my_cross_val(log_reg, 'err_rate', X_test, y_test)
# print error rate on test data
print("Error Rates with Best eta", best_log_reg_err)

# plot the loss function value for each iteration of SGD
import matplotlib.pyplot as plt
plotv = log_reg_best.fit(X_train, y_train)
plt.plot(plotv,color='red')
plt.title("Loss Function Value for Each Iteration of SGD")
plt.xlabel("Iteration number")
plt.ylabel("Loss Function Value")