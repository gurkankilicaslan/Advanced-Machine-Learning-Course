import numpy as np

class MyPCA():
    
    def __init__(self, num_reduced_dims):
        self.num_reduced_dims = num_reduced_dims

    def fit(self, X):
        # cov = X.T @ X
        
        # cov = np.cov(X)
        # cov = np.cov(X, rowvar=False)
        
        cov = (1/(X.shape[0] - 1)) * X.T @ X
        # print(cov.shape)
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # eigenvalues, eigenvectors = scipy.linalg.eigh(cov)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        N = self.num_reduced_dims
        modif_eigenvectors = eigenvectors[:, -N:]
        self.modif_eigenvectors = modif_eigenvectors

    def project(self, x):
        coordinates = (self.modif_eigenvectors.T @ x.T).T
        return coordinates
