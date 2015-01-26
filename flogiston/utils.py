import numpy as np

def protect_zeroes(my_input, threshold=1e-10):
    return np.where(my_input > threshold, my_input, np.ones_like(my_input) * threshold)
    
