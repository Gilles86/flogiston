from scipy import stats
import numpy as np
from sampler import Node
from base import get_likelihood_hrf


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

def likelihood_later_bold(rts, bold, frametimes, onsets, ter, v, sv, b, intercept_height, intercept_delay, theta_height, theta_delay):
    
    d = rts - ter

    sampled_vs = b / d
    
    heights = intercept_height + theta_height * (sampled_vs - v)    
    delays = intercept_delay + theta_delay * (sampled_vs - v)
    
    return get_likelihood_hrf(bold, frametimes, onsets, heights, peak_delays=delays)
    
    
    


class SingleLaterBOLD(Node):
    
    def __init__(self, name, ter, v, b, sv=1.0, intercept_height=0.0, intercept_delay=6.0, theta_height=0.0, theta_delay=0.0, value=None):
        
        self.stochastic = False
        Node.__init__(self, name, {'ter':ter, 
                                                     'v':v, 
                                                     'b':b, 
                                                     'sv':sv, 
                                                     'intercept_height':intercept_height,
                                                     'intercept_delay':intercept_delay,
                                                     'theta_height':theta_height,
                                                     'theta_delay':theta_delay}, value=value)
        
    
    def _likelihood(self, values):
        return likelihood_later_bold(values[0], values[1], values[2], values[3],
                                self.parents['ter'].get_value(),
                                self.parents['v'].get_value(),
                                self.parents['sv'].get_value(),
                                self.parents['b'].get_value(),
                                self.parents['intercept_height'].get_value(),
                                self.parents['intercept_delay'].get_value(),
                                self.parents['theta_height'].get_value(),
                                self.parents['theta_delay'].get_value(),)
    
   
 
class SingleLater(Node):
    
    def __init__(self, name, ter, v, b, sv=1.0, value=(None, None)):
        
        self.stochastic = False

        Node.__init__(self, name, {'ter':ter, 'v':v, 'b':b, 'sv':sv}, value=value )
        
    
    def _likelihood(self, values):
        return likelihood_later_single(self.value, 
                                self.parents['ter'].get_value(),
                                self.parents['v'].get_value(),
                                self.parents['sv'].get_value(),
                                self.parents['b'].get_value(),)
    

def likelihood_later_double(rts, ters, vs, svs, bs):

    """ response 1 won """ 
    D = b1 / (rts - ter1)
    
    ll = (D**2 / b1) * \
          stats.truncnorm.cdf(a=(0-v2)/float(sv2), b=np.infty, loc=v2, scale=sv2, x=D)  * \
          stats.truncnorm.pdf(a=(0-v1)/float(sv1), b=np.infty, loc=v1, scale=sv1, x=D)

    ll = np.sum(np.log(ll))
    
    return ll


class SingleLater(Node):
    
    def __init__(self, name, ter, v, b, sv=1.0, value=(None, None)):
        
        self.stochastic = False

        Node.__init__(self, name, {'ter':ter, 'v':v, 'b':b, 'sv':sv}, value=value )
        
    
    def _likelihood(self, values):
        return likelihood_later_single(self.value, 
                                self.parents['ter'].get_value(),
                                self.parents['v'].get_value(),
                                self.parents['sv'].get_value(),
                                self.parents['b'].get_value(),)
