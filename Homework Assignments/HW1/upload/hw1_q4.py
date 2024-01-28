################################
# DO NOT EDIT THE FOLLOWING CODE
################################
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Lasso
import numpy as np

from MyRidgeRegression import MyRidgeRegression
from my_cross_val import my_cross_val

# load dataset
X, y = fetch_california_housing(return_X_y=True)

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

# Split dataset into train and test sets
NUM_TRAIN = int(np.ceil(num_data*0.8))
NUM_TEST = num_data - NUM_TRAIN

X_train = X[:NUM_TRAIN]
X_test = X[NUM_TRAIN:]
y_train = y[:NUM_TRAIN]
y_test = y[NUM_TRAIN:]

lambda_vals = [0.01, 0.1, 1, 10, 100]

#####################
# ADD YOUR CODE BELOW
#####################

rid_mean = []
rid_std = []

las_mean = []
las_std = []
for lambda_val in lambda_vals:
    # print("\n")
    print("\n" + " for lambda = " + str(lambda_val) + " :")
    # instantiate ridge regression object
    rid_reg = MyRidgeRegression(lambda_val)
    
    # call to your CV function to compute mse for each fold
    rid_reg_mse = my_cross_val(rid_reg, 'mse', X, y)
    # print mse from CV
    print("MSE for Ridge Regression", rid_reg_mse)
    
    # instantiate lasso object
    las = Lasso(lambda_val)
    # call to your CV function to compute mse for each fold
    las_mse = my_cross_val(las, 'mse', X, y)
    # print mse from CV
    print("MSE for Lasso Regression", las_mse)
    
    #  Finding best lambda for Ridge Regression:
    rid_list = []
    for j in range(len(rid_reg_mse) - 1):
        rid_list.append(float(rid_reg_mse[j]))
    rid_mean_by_lambda = np.mean(rid_list)
    rid_std_by_lambda = np.std(rid_list)
    
    rid_mean.append(rid_mean_by_lambda)
    rid_std.append(rid_std_by_lambda)
    idx_min_rid_mean = rid_mean.index(min(rid_mean))
    best_rid_lambda = lambda_vals[idx_min_rid_mean]
    
    #  Finding best lambda for Lasso Regression:
    las_list = []
    for j in range(len(las_mse) - 1):
        las_list.append(float(las_mse[j]))
    las_mean_by_lambda = np.mean(las_list)
    las_std_by_lambda = np.std(las_list)
    
    las_mean.append(las_mean_by_lambda)
    las_std.append(las_std_by_lambda)
    idx_min_las_mean = las_mean.index(min(las_mean))
    best_las_lambda = lambda_vals[idx_min_las_mean]
    
print("\n")
print("Best lambda for Ridge Regression is ", best_rid_lambda)
print("Best lambda for Lasso Regression is ", best_las_lambda)

# instantiate ridge regression and lasso objects for best values of lambda
rid_reg_best = MyRidgeRegression(best_rid_lambda)
las_best = Lasso(best_las_lambda)
# fit models using all training data
rid_reg_best.fit(X_train, y_train)
las_best.fit(X_train, y_train)
# predict on test data
rid_reg_best.predict(X_test)
las_best.predict(X_test)
# compute mse on test data
best_rid_reg_mse = my_cross_val(rid_reg_best, 'mse', X_test, y_test)
best_las_mse = my_cross_val(las_best, 'mse', X_test, y_test)
# print mse on test data
print("MSE for Ridge Regression for Best Lambda", best_rid_reg_mse)
print("MSE for Lasso Regression for Best Lambda", best_las_mse)

