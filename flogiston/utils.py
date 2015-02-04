import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

## NOTE: is it a good idea to set a hard numerical threshold (i.e., 7)?
##  should maybe depend on mean/sd? E.g. crit=mean+7*sd ?
def pnormP(x, mean=0, sd=1):
    """standard normal CDF with numerical stability
    R: pnormP  <- function(x,mean=0,sd=1,lower.tail=T){ifelse(abs(x)<7,pnorm(x,mean,sd,lower.tail),ifelse(x<0,0,1))}
    """
    return np.where(np.abs(x-mean)<7.*sd, stats.norm.cdf(x, loc=mean,scale=sd), np.where(x<mean,0,1))

def dnormP(x, mean=0, sd=1):
    """standard normal PDF with numerical stability
    R: dnormP <- function(x,mean=0,sd=1,lower.tail=T){ifelse(abs(x)<7,dnorm(x,mean,sd),0)}
    """
    return np.where(np.abs(x-mean)<7.*sd,stats.norm.pdf(x, loc=mean, scale=sd),0)


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


def data_histogram(responses, RTs, t_max=2.0, bins=None, color_palette='Set1', **kwargs):
    
    if 'alpha' not in kwargs:
        kwargs['alpha'] = 0.8

    if bins == None:
        bins = np.linspace(0, t_max, np.min((np.max((responses.shape[0] / 25, 25)), 50)))

    bin_width = bins[1] - bins[0]

    colors = sns.color_palette(color_palette)

    for resp, color in zip(np.unique(responses), colors):
        hist, bin_edges = np.histogram(RTs[responses == resp], bins=bins)
        hist = hist / bin_width / responses.shape[0]
        plt.bar(bin_edges[:-1], hist, width=(bin_edges[1] - bin_edges[0]), color=color, **kwargs)
        
def pack_conditions_responses_param(param, n_conditions, n_responses):
        if type(param) != list:
            param = [[param] * n_conditions] * n_responses
        
        return param
