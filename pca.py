'''
Author: Ambareesh Ravi
Date: 08-02-2020
Title: PCA

Links and References:
    Datasets:
        -MNIST
        
    Papers:
        -
        
    Lectures, tutorials, articles and posts:
        - 
        
    Additional comments / notes:
        - 
'''

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, PoV = 1.0, n_components = 0, method = 'SVD', isSVD_dual = False):
        '''
        Initializes parameters for PCA
        
        ## INPUT X should be of shape (samples, features) ##
        
        Args:
            PoV - Proportion of Variance as <float>
            n_components - reduced dimensions <int>
            method - SVD / EIG
            singular_type - left / right
        '''
        self.PoV = PoV
        self.n_components = n_components
        self.method = method
        self.isSVD_dual = isSVD_dual
        
    def adjust_mean(self, X):
        
        '''
        Function to adjust data zero-mean
        
        Args:
            X - input data <np.array>
        Returns:
            -
        Exception:
            -
        '''
        return X - np.mean(X.T, axis=1)
    
    def get_covariance_matrix(self):
        '''
        Function to calculate the covariance matrix
        
        Args:
            -
        Returns:
            -
        Exception:
            -
        '''
        self.Xm = self.adjust_mean(self.X)
        return np.matmul(self.Xm.T, self.Xm)/(self.Xm.shape[0]-1) # conj() if complex values in data
    
    def getEig(self, A):
        '''
        Function to get Eigen Value and Eigen vectors
        
        Args:
            A - Input matrix <np.array>
        Returns:
            Eigen Value, Eigen vectors as <np.array>
        Exception:
            -
        '''
        return np.linalg.eig(A) # returns eigen values and eigen vectors
    
    def getSVD(self, A):
        '''
        Function to get Singluar Value Decomposition
        
        Args:
            A - Input matrix <np.array>
        Returns:
            U, D, V as <np.array>
        Exception:
            -
        '''
        # compute right singluar values
        R, V = np.linalg.eig(np.matmul(A.T, A))
        # sort in decreasing order
        sort_indices = np.argsort(R)[::-1]
        R = R[sort_indices]
        V = V[:,sort_indices].T

        # compute left singluar values
        L, U = np.linalg.eig(np.matmul(A, A.T))
        # sort in decreasing order
        sort_indices = np.argsort(L)[::-1]
        L = L[sort_indices]
        U = U[:,sort_indices]
        
        #find sigma with lowest dimension (singular values)
        if L.shape[-1] < R.shape[-1]:
            D = np.array([np.sqrt(i) for i in L])
        else:
            D = np.array([np.sqrt(i) for i in R])
        return U, D, V
    
    def sortEig(self):
        '''
        Function to sort eigen vectors based on decreasing eigen values
        
        Args:
            -
        Returns:
            -
        Exception:
            -
        '''
        idx = np.argsort(self.eigenValues)[::-1]
        self.eigenVectors = self.eigenVectors[:, idx]
        self.eigenValues = self.eigenValues[idx]
    
    def calcPoV(self, ev, stop_idx):
        '''
        Function to calculate the proportion of variance
        
        Args:
            Eigen values <np.array>, Eigen Vectors <np.array>
        Returns:
            PoV <float>
        Exception:
            -
        '''
        return np.sum(ev[:stop_idx]) / np.sum(ev)
    
    def setPoV(self, ev):
        '''
        Function to set PoV to operate on
        
        Args:
            -
        Returns:
            -
        Exception:
            -
        '''
        count_idx = 1
        while self.calcPoV(ev, count_idx) < self.PoV:
            count_idx += 1
        return count_idx
    
    def get_basis(self, eVectors, index):
        '''
        Function to set basis
        
        Args:
            index - reduced dimension <int>
        Returns:
            eigen vectors for reduced dimensions as <np.array>
        Exception:
            -
        '''
        return eVectors[:, :index] # eigen vectors stacked as columns each
    
    def isSimilar(self, m1, m2):
        
        '''
        Function to check similarity between two arrays
        
        Args:(784, 784)
            m1, m2 as <np.array>
        Returns:
            <bool> on similarity
        Exception:
            -
        '''
        return np.testing.assert_almost_equal(m1,m2)

    def plot_eigenValues(self, idx = -1):
        '''
        Function to plot eigen values as a scree plot
        
        Args:
            -
        Returns:
            -
        Exception:
            -
        '''
        x_pos = list(range(len(self.eigenValues[:idx])))
        plt.figure(figsize= (10,10))
        plt.bar(x = x_pos, height = self.eigenValues[:idx], color = 'red')
        plt.xlabel("Number of Components")
        plt.ylabel("Eigen Value")
        plt.title("Eigen Value vs n_components")
        plt.show()
           
    def fit_EIG(self):
        '''
        Function to fit input data to PCA EIG method
        
        Args:
            -
        Returns:
            -
        Exception:
            -
        '''
        self.eigenValues, self.eigenVectors = self.getEig(self.cov) # each column an eigen vector
        self.sortEig()

        if self.PoV < 1:
            newPoV_index = self.setPoV(self.eigenValues)
            print("Setting number of components to", newPoV_index, "for PoV", self.PoV)
            self.basis = self.get_basis(self.eigenVectors, newPoV_index)
        elif self.n_components > 0 and self.n_components < min(self.X.shape):
            self.basis = self.get_basis(self.eigenVectors, self.n_components)
        else:
            self.basis = self.eigenVectors
        
    def transform_EIG(self, Xm_new):
        '''
        Function to transform mean shifted new inputs for EIG method
        
        Args:
            Xm_new - input <np.array>
        Returns:
            -
        Exception:
            -
        '''
        return self.basis.T.dot(Xm_new.T).T

    def reconstruct_EIG(self, reduced_X):
        '''
        Function to reconstruct reduced data to original dimensionality using EIG
        
        Args:
            reduced_X - reduced data <np.array>
        Returns:
            reconstructed / recovered data <np.array>
        Exception:
            -
        '''
        self.recovery = np.dot(self.basis, reduced_X.T).T # recovery
        self.recovery += self.recovery.mean(axis = 0) # mean shift
        return self.recovery
    
    def fit_SVD(self):
        '''
        Function to fit input data to PCA SVD method
        
        Args:
            -
        Returns:
            -
        Exception:
            -
        '''
        # original X is of dimension (n,d)
        self.Xm = self.adjust_mean(self.X)
        self.U, self.D, self.V = self.getSVD(self.Xm.T) # convert to (d,n)
        self.U_org, self.V_org = self.U, self.V 
        
        self.Sigma = np.zeros_like(self.Xm.T)
        for idx, i in enumerate(self.D):
            self.Sigma[idx][idx] = i
        self.Sigma_org = self.Sigma.copy()
        
        # considering U to have dim (d,d) and so taking it as a basis
        if self.PoV < 1:
            newPoV_index = self.setPoV(self.D)
            print("Setting number of components to", newPoV_index, "for PoV", self.PoV)
            self.U = self.get_basis(self.U, newPoV_index)
            if self.isSVD_dual: self.Sigma = self.get_basis(self.Sigma.T, newPoV_index).T
        elif self.n_components > 0 and self.n_components < min(self.X.shape):
            self.U = self.get_basis(self.U, self.n_components)
            if self.isSVD_dual: self.Sigma = self.get_basis(self.Sigma.T, self.n_components).T

    def transform_SVD(self, Xm_new):
        '''
        Function to transform mean shifted new inputs for SVD method
        
        Args:
            Xm_new - input <np.array>
        Returns:
            -
        Exception:
            -
        '''
        # .T in the result for for the shape (n,p)
        if self.isSVD_dual:
            # U = X V inv(Sigma)
            # Y = U.T x
            return np.matmul(np.matmul(self.Xm.T, np.matmul(self.V, np.linalg.pinv(self.Sigma))).T, Xm_new.T).T
        else:
            # Y = U.T x
            return np.matmul(self.U.T, Xm_new.T).T 
    
    def reconstruct_SVD(self, reduced_X):
        '''
        Function to reconstruct reduced data to original dimensionality using SVD
        
        Args:
            reduced_X - reduced data <np.array>
        Returns:
            reconstructed / recovered data <np.array>
        Exception:
            -
        '''
        # .T in result for final dim to be (n,d)
        if self.isSVD_dual:
            # X_r = U U.T X
            # X_r = (XV inv(Sigma))(XV inv(Sigma)).T X
            # U = X V inv(Sigma)
            term_U = np.matmul(self.Xm.T, np.matmul(self.V, np.linalg.pinv(self.Sigma)))
            self.recovery = np.matmul(term_U, reduced_X.T).T
        else:
            self.recovery = np.matmul(self.U, reduced_X.T).T 
        self.recovery += np.mean(self.X.T, axis=1)
        return self.recovery
    
    def fit(self, X):
        
        '''
        Function to fit input data to PCA
        
        Args:
            X - input data (n,d) as <np.array>
        Returns:
            -
        Exception:
            -
        '''
        # original X is of dimension (n,d)
        self.X = X
        n,d = self.X.shape
        
        # choose SVD method based on the input
        if n < d: self.isSVD_dual = True
            
        if self.n_components == 0: self.n_components = min(X.shape)
        self.cov = self.get_covariance_matrix()
        if self.method == "EIG":
            self.fit_EIG()
        elif self.method == "SVD":
            self.fit_SVD()

    def transform(self, Xnew):
        '''
        Function to transform new inputs with respect to basis
        
        Args:
            Xnew - input <np.array>
        Returns:
            -
        Exception:
            -
        '''
        Xm_new = self.adjust_mean(Xnew)
        if self.method == "EIG":
            return self.transform_EIG(Xm_new).real
        elif self.method == "SVD":
            return self.transform_SVD(Xm_new).real
    
    def fit_transform(self, X):
        '''
        Function to fit and transform new inputs with respect to basis
        
        Args:
            X - input <np.array>
        Returns:
            transformed / reduced data <np.array>
        Exception:
            -
        '''
        if self.method == "EIG":
            self.fit(X)
            return self.transform(X)
        elif self.method == "SVD":
            self.fit(X)
            return self.transform(X)
    
    def reconstruct(self, reduced_X):
        '''
        Function to reconstruct reduced data to original dimensionality
        
        Args:
            reduced_X - reduced data <np.array>
        Retositurns:
            reconstructed / recovered data <np.array>
        Exception:
            -
        '''
        if self.method == "EIG":
            return self.reconstruct_EIG(reduced_X).real
        elif self.method == "SVD":
            return self.reconstruct_SVD(reduced_X).real
        
    def visualize_reconstruction(self, original, reconstructed, title):
        fig, ax = plt.subplots(1,2)
        plt.suptitle(title)
#         plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        ax[0].imshow(original)
        ax[0].set_title('Original')
        ax[1].imshow(reconstructed)
        ax[1].set_title('Reconstructed')
        plt.show()
                
if __name__ == '__main__':
    # Load data
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = np.array([i.flatten() for i in X_train])
    X_test = np.array([i.flatten() for i in X_test])
    
    X, y = X_train, y_train
    
    # Do PCA
    mypca = PCA(n_components = 100, method = "EIG", isSVD_dual = False)
    mypca.fit(X)
    reduced_X = mypca.transform(X)
    recons = mypca.reconstruct(reduced_X)
    
    # Check the results
    n_checks = 2
    n_checks = [np.random.randint(0, len(X)) for _ in range(n_checks)] # list of random indices
    for rand_idx in n_checks:
        mypca.visualize_reconstruction(np.reshape(X[rand_idx], (28,28)), np.reshape(recons[rand_idx], (28,28)), "Result of random pick: "+ str(rand_idx))