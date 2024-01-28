import numpy as np

class MyRidgeRegression():

    def __init__(self, lambda_val):
        self.lambda_val = lambda_val
    
    def fit(self, X, y):
        self.w = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X) + self.lambda_val * np.identity(X.shape[1])), X.T), y)
    
    def predict(self, X):
        yHat = np.dot(X, self.w)
        return yHat


