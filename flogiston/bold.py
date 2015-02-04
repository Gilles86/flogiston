from scipy import stats
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

def simulate_bold(frametimes, onsets, heights, peak_delays=None):
    
    if peak_delays == None:
        peak_delays = 6

    peak_disp =1

    bold =stats.gamma.pdf(frametimes[:,np.newaxis],
                         peak_delays / peak_disp,
                         loc=onsets[np.newaxis, :],
                         scale = peak_disp)
    
    bold *= heights
    
    bold = bold.sum(1)
    
    return bold
    
    
def get_likelihood_hrf(y, frametimes, onsets, heights, peak_delays=None, plot=False):
    
    y_ = simulate_bold(frametimes, onsets, heights, peak_delays)
    
    resid = y - y_

    if plot:
        plt.plot(y)
        plt.plot(y_)
        plt.xlim(0, 50)

    return pm.normal_like(resid, 0, 1/np.std(resid)**2)

