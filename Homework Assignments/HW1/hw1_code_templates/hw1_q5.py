################################
# DO NOT EDIT THE FOLLOWING CODE
################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_cross_val import my_cross_val
from MyLDA import MyLDA

# load dataset
data = pd.read_csv('hw1_q4_dataset.csv', header=None).to_numpy()
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

for lambda_val in lambda_vals:

    # instantiate LDA object

    # call to your CV function to compute error rates for each fold

    # print error rates from CV

# instantiate LDA object for best value of lambda

# fit model using all training data

# predict on test data

# compute error rate on test data

# print error rate on test data

