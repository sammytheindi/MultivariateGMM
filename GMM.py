import numpy as np
from KMeans import KMeans

class GMM():
    def __init__(self):
        self.kmeans_cls_ = KMeans()
        self.means_ = None
        self.cov_ = None
        self.mixture_weights_ = None
        self.membership_weights_ = None
        self.k_ = None
        self.ll_graph_ = []
    
    def fit(self, X, k, tol_=1e-6):
        self.k_ = k
        self.initialize(X)
        new_ll = self.get_log_likelihood(X)
        old_ll = new_ll - tol_*10
        while old_ll - new_ll < -tol_:
            self.ll_graph_.append(new_ll)
            self.gaussian_probabilities_multiple(X, normalized=True)
            self.update_mixture_weights()
            self.update_means(X)
            self.update_var(X)
            old_ll = new_ll
            new_ll = self.get_log_likelihood(X)
    
    def initialize(self, X):
        self.kmeans_cls_.fit(X, self.k_)
        self.means_ = self.kmeans_cls_.means_
        self.cov_ = self.kmeans_cls_.cov_
        self.mixture_weights_ = (np.array([1]*self.k_))/self.k_
        
    def gaussian_probabilities_multiple(self, X, normalized=True):
        d = X.shape[-1]

        input_ = X[:, None, :]

        exp_part = -0.5*np.einsum('ijk,jkl,ijl->ij', input_-self.means_, np.array(list(map(np.linalg.inv, self.cov_))), input_-self.means_)
        output = (1/((2*np.pi)**(d/2)*np.array(list(map(lambda x: np.linalg.det(x)**(1/2), self.cov_)))))[None, :]*np.exp(exp_part)

        if normalized:
            output = np.einsum('ij,j->ij', output, self.mixture_weights_)
            output = output/np.sum(output, axis = 1, keepdims=True)
            self.membership_weights_ = output
        else:
            return output
        
    def update_mixture_weights(self):
        self.mixture_weights_ = np.einsum('ij->j', self.membership_weights_)/self.membership_weights_.shape[0]
    
    def update_means(self, X):
        self.means_ = np.einsum('id,ik->kd', X, self.membership_weights_)/np.einsum('ik->k', self.membership_weights_)[:, None]
    
    def update_var(self, X):
        input_ = X[:, None, :]
        self.cov_ = np.einsum('ij,ijk,ijl->jlk',self.membership_weights_, input_-self.means_, input_-self.means_)/np.einsum('ik->k', self.membership_weights_)[:, None, None]
    
    def get_log_likelihood(self, X):
        output = self.gaussian_probabilities_multiple(X, normalized=False)
        output = np.log(np.einsum('ij,j->i', output, self.mixture_weights_))
        return np.einsum('i->', output)
