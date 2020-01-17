import numpy as np

class KMeans():
    def __init__(self):
        self.k = None
        self.means_ = None
        self.cov_ = None
        self.memberships_ = None
    
    def fit(self, X, k, tol_=1e-3):
        self.k = k
        self.initialize(X)
        means_old = 0
        while np.linalg.norm(self.means_ - means_old, axis = 1).mean() > tol_:
            means_old = self.means_
            self.update_memberships(X)
            self.update_means(X)
        self.update_covariances(X)
        
    def initialize(self, X):
        self.means_ = X[np.random.permutation(len(X))[:self.k]]
    
    def update_memberships(self, X):
        self.memberships_ = np.argmin(np.linalg.norm(X[:,None,:]-self.means_, axis=-1), axis=-1)
    
    def update_means(self, X):
        self.means_ = np.array([X[self.memberships_==k].mean(axis=0) for k in range(self.k)])
    
    def update_covariances(self, X):
        self.cov_ = np.array([np.cov(X[self.memberships_==k].T) for k in range(self.k)])
