from scipy import stats
import numpy as np


def likelihood_later_single(rts, ter, v, sv, b):
    
    D = b / (rts - ter)
    
    ll = (D**2 / b) * stats.truncnorm.pdf(a=(0-v)/float(sv), b=np.infty, loc=v, scale=sv, x=D)
    ll = np.sum(np.log(ll))
    
    return ll

    
def likelihood_later_double(rts, ter1, ter2, v1, v2, sv1, sv2, b1, b2):

    """ response 1 won """ 
    D = b1 / (rts - ter1)
    
    ll = (D**2 / b1) * \
          stats.truncnorm.cdf(a=(0-v2)/float(sv2), b=np.infty, loc=v2, scale=sv2, x=D)  * \
          stats.truncnorm.pdf(a=(0-v1)/float(sv1), b=np.infty, loc=v1, scale=sv1, x=D)

    ll = np.sum(np.log(ll))
    
    return ll

    
