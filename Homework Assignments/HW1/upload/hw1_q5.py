################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_cross_val import my_cross_val
from MyLDA import MyLDA

# load dataset
data = pd.read_csv('hw1_q5_dataset.csv', header=None).to_numpy()
X = data[:,:-1]
y = data[:,-1]

num_data, num_features = X.shape

plt.scatter(X[:1000, 0], X[:1000, 1])
plt.scatter(X[1000:, 0], X[1000:, 1])
plt.show()

# shuffle dataset
np.random.seed(2023)
perm = np.random.permutation(num_data)

X = X.tolist()
y = y.tolist()

X = [X[i] for i in perm]
y = [y[i] for i in perm]

X = np.array(X)
y = np.array(y)

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
LDA_mean = []
LDA_std = []

lambda_vals = [-0.008, -0.007, -0.006, -0.005, -0.004, -0.003, -0.002, -0.001, 0]
# print(lambda_vals)
for lambda_val in lambda_vals:
    # print("\n")
    print("\n" + " for lambda = " + str(lambda_val) + " :")
    # instantiate LDA object
    LDA = MyLDA(lambda_val)
    # print(type(LDA))
    # call to your CV function to compute error rates for each fold
    errLDA = my_cross_val(LDA, 'err_rate', X, y)
    # print error rates from CV
    print ("Error Rates for LDA", errLDA)
    
    #  Finding best lambda for LDA:
    LDA_list = []
    for j in range(len(errLDA) - 1):
        LDA_list.append(float(errLDA[j]))
    LDA_mean_by_lambda = np.mean(LDA_list)
    LDA_std_by_lambda = np.std(LDA_list)
    
    LDA_mean.append(LDA_mean_by_lambda)
    LDA_std.append(LDA_std_by_lambda)
    idx_min_LDA_mean = LDA_mean.index(min(LDA_mean))
    best_rid_lambda = lambda_vals[idx_min_LDA_mean]
    

X_train = pd.DataFrame(X_train)


# instantiate LDA object for best value of lambda
best_LDA = MyLDA(best_rid_lambda)
# fit model using all training data


best_LDA.fit(X_train, y_train)
# predict on test data
best_LDA.predict(X_test)
# compute error rate on test data
best_errLDA = my_cross_val(LDA, 'err_rate', X_test, y_test)
# print error rate on test data
print("\n")
print("Best lambda for LDA is ", best_rid_lambda)
print ("Error Rates for LDA for Best Lambda", best_errLDA)
