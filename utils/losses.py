# Author: Ambareesh Ravi
# Date: 21-01-2020
# Title: losses
# Description: Module containing all the necessary loss functions

import numpy
import math


def mean_absolute_error(actual, predicted):
    '''
    Description:
        Function to calculate Mean Absolute Error
        https://en.wikipedia.org/wiki/Mean_absolute_error
    Args:
        actual as <numpy array>
        predicted as <numpy array>
    Returns:
        MAE value
    Exception:
        General Exception
    '''
    
    assert len(actual)==len(predicted)
    return (sum([abs(a-p) for a, p in zip(actual.tolist(), predicted.tolist())]))/len(actual)

def mean_squared_error(actual, predicted):
    '''
    Description:
        Function to calculate Mean Square Error
        https://en.wikipedia.org/wiki/Mean_squared_error
    Args:
        actual as <numpy array>
        predicted as <numpy array>
    Returns:
        MSE value
    Exception:
        General Exception
    '''
    
    assert len(actual)==len(predicted)
    return (sum([(a-p)**2 for a, p in zip(actual.tolist(), predicted.tolist())]))/len(actual)
