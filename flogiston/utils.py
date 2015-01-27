import numpy as np
from scipy import stats

def shuffle(value):
#    a = value
    np.random.shuffle(value)
    return value

def protect_zeroes(my_input, threshold=1e-10):
    return np.where(my_input > threshold, my_input, np.ones_like(my_input) * threshold)


def simple_single_trial_ols(frametimes, onsets, bold):
    
    X =stats.gamma.pdf(frametimes[:,np.newaxis],
                     6.0,
                     loc=onsets[np.newaxis, :],
                     scale =1.0)
    
    return np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(bold)
    
    
