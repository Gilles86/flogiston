from scipy import stats
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from sampler import Node

def simulate_bold(frametimes, onsets, heights, peak_delays=None, peak_disp=None, undershoot=False, ):
    
    if peak_delays == None:
        peak_delays = 6
    
    if peak_disp == None:
        peak_disp =1

    #print onsets, onsets.shape
    #print onsets[np.newaxis, :]

    bold =stats.gamma.pdf(frametimes[:,np.newaxis],
                         peak_delays / peak_disp,
                         loc=onsets[np.newaxis, :],
                         scale = peak_disp)
    
    if undershoot:
        under_delay = 16
        under_disp = 1
        under_ratio = 6
        us = stats.gamma.pdf(frametimes[:,np.newaxis],
                                   under_delay / under_disp,
                                   loc=0,
                                   scale = under_disp)

        bold -= us / under_ratio
    
    bold *= heights
    
    bold = bold.sum(1)
    
    return bold
    
    
def get_likelihood_hrf(y, frametimes, onsets, heights, dispersions=None, peak_delays=None, undershoot=True, plot=False):
    
    #print onsets
    #print y, frametimes, onsets, heights, peak_delays

    #print onsets, heights, peak_delays, dispersions
    y_ = simulate_bold(frametimes, onsets, heights, peak_delays, dispersions, undershoot=undershoot)
    
    resid = y - y_

    if plot:
        plt.plot(y)
        plt.plot(y_)
        plt.xlim(0, 50)

    return pm.normal_like(resid, 0, 1/(resid.T.dot(resid)/(len(resid) - 1)))



class BOLD(Node):

    
    def __init__(self, name, heights, delays, dispersions, conditions, frametimes, onsets, value=None, **kwargs):
        " This Node expects lists of nodes as parents!!"
        " Example: [[ter_resp1_cond1, ter_resp1_cond2], [ter_resp2_cond1, ter_resp2_cond2]] "
        
        self.stochastic = False
        self.children = []
        
        self.n_conditions = np.max(conditions)

        parents = {}
        for condition in np.arange(1, self.n_conditions+1):
            parents['height_%d' % condition] = heights[condition - 1]
            parents['delay_%d' % condition] = delays[condition - 1]
            parents['dispersion_%d' % condition] = dispersions[condition - 1]

        self.conditions = conditions
        self.frametimes = frametimes
        self.onsets = onsets

        Node.__init__(self, name, parents, value=value, **kwargs )

    
    def _likelihood(self, value):
        
        heights =  np.zeros_like(self.onsets)
        delays = heights.copy()
        dispersions = heights.copy()

        for condition in np.arange(1, self.n_conditions+1):
            heights[self.conditions == condition] = self.parents['height_%d' % condition].get_value()
            delays[self.conditions == condition] = self.parents['delay_%d' % condition].get_value()
            dispersions[self.conditions == condition] = self.parents['dispersion_%d' % condition].get_value()

        return get_likelihood_hrf(value, self.frametimes, self.onsets, heights, peak_delays=delays, dispersions=dispersions)

def plot_traces(heights, delays, dispersions=None, **kwargs):
    
    alpha = kwargs.pop('alpha', 0.2)
    color = kwargs.pop('color', 'k')

    if dispersions == None:
        dispersions = np.ones_like(heights)

    t = np.linspace(0, 20)

    for h, delay, dispersion in zip(heights, delays, dispersions):
        plt.plot(t, simulate_bold(t, np.array([0]), np.array([h]), np.atleast_1d(delay), np.atleast_1d(dispersion), undershoot=True),
                 alpha=alpha,
                 color=color) 


