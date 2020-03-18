'''
Author: Ambareesh Ravi
Date: 7-02-2020
Title: ML, MAP

Links and References:
    Datasets:
        -
        
    Papers:
        -
        
    Lectures, tutorials, articles and posts:
        - 
        
    Additional comments / notes:
        - 
'''

# Imports
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from time import time
import seaborn as sb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error

class ML_MAP:
    def __init__(self):
        '''
        Initializes the variables for Question 2
        '''
        pass
        
    def ML(self, x, mean, cov, Pc = None):
        '''
        Function to evaluate function g(x) for Maximum Likelihood Estimation
        
        Args:
            x - input np.array (1,d)
            cov - covariance matrix <np.array>
            mean - mean vector - <np.array>
            Pc - Prior Value <float>
        Returns:
            value at x
        Exception:
            -
        '''
        # ML g(x)
        return (1/(np.power((2*np.pi), (x.shape[-1]/2)) * np.sqrt(np.linalg.det(cov)))) * np.exp((-1/2) * np.matmul(np.matmul((x - mean).T, np.linalg.inv(cov)), (x - mean)))

    def MAP(self, x, mean, cov, Pc):
        '''
        Function to evaluate function g(x) for Maximum A Posteriori
        
        Args:
            x - input np.array (1,d)
            cov - covariance matrix <np.array>
            mean - mean vector - <np.array>
            Pc - Prior Value <float>
        Returns:
            value at x
        Exception:
            -
        '''
        # MAP g(x)
        return np.power(np.linalg.det(cov), -0.5) * np.exp((-1/2) * np.matmul(np.matmul((x - mean).T, np.linalg.inv(cov)), (x - mean))) * Pc