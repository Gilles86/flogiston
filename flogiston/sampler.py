import numpy as np
from scipy import stats
from utils import protect_zeroes, shuffle
from progressbar import ProgressBar
from collections import OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt



from copy import deepcopy, copy

class Model(object):
    
    def __init__(self, nodes):

        self.nodes = nodes        
        
        stochastics = []

        for node in nodes:
            stochastics += [n for n in node.parents.values() if n.is_stochastic]
        
        self.stochastics = OrderedDict([(n.name, n) for n in set(stochastics)])        
             
    def get_node(self, name):
        return self.nodes[name]
    
    def copy(self):
        return deepcopy(self)
    
    def get_starting_values(self):
        for node in self.stochastics.values():
            node.get_starting_values()
            
    def get_param_vector(self):
        return np.array([node.get_value() for node in self.stochastics.values()])
    
    def set_param_vector(self, vector):
        
        for i, node in enumerate(self.stochastics.values()):
            node.set_value(vector[i])
            
    def get_logp(self):
        logp = 0
        for node in self.nodes:
            logp += node.get_logp()

        return logp
            
class DESampler(object):
    
    
    def __init__(self, model):
        self.model = model
        
    
    def sample(self, n_chains=None, n_samples=500, sampling_scheme=None, gamma=None, b=0.01):
        
        
        n_params = len(self.model.stochastics)
        
        if not n_chains:
            n_chains = 2*n_params + 1
            
        if not gamma:
            gamma = 2.38 / np.sqrt(2 * n_params)
        
        self.chains = np.zeros((n_chains, n_params, n_samples))
        self.accepted = np.zeros_like(self.chains)
        self.new_logp = np.zeros((n_chains, n_samples))
        self.old_logp = np.zeros((n_chains, n_samples))
        
        models= []
        
        for i in np.arange(n_chains):
            models.append(self.model.copy())
            models[-1].get_starting_values()
        
        for chain in np.arange(n_chains):
            self.chains[chain, :, 0] = models[chain].get_param_vector()
            
        selects = np.apply_along_axis(shuffle, 1, np.tile(np.arange(n_chains), (n_chains*n_samples, 1)))[:, :2]
        selects = selects.reshape((n_chains, n_samples, 2))
        
        bs = stats.uniform.rvs(-b, b*2, (n_chains, n_params, n_samples))
        
        unif = np.log(stats.uniform.rvs(0, 1, (n_chains, n_samples)))

        pbar = ProgressBar(n_samples)


        if sampling_scheme:
            tmp = []
            for block in sampling_scheme:
                tmp.append([self.model.stochastics.keys().index(e) for e in block])
        
            sampling_scheme = tmp
        
        for sample in np.arange(1, n_samples):


            if sampling_scheme:
                for block in sampling_scheme:
                    for chain in np.arange(n_chains):
                        
                        old_logp = models[chain].get_logp()
                        
                        chain_m = selects[chain, sample, 0]
                        chain_n = selects[chain, sample, 1]

                        current_param = models[chain].get_param_vector()

                       
                        proposal_param = current_param
                        proposal_param[block] = current_param[block] + gamma * (self.chains[chain_m, :, sample-1] - self.chains[chain_n, :, sample-1])[block] + bs[chain, :, sample][block]
                        
                        tmp_model = models[chain].copy()
                        tmp_model.set_param_vector(proposal_param)
                        
                        new_logp = tmp_model.get_logp()
                        
                        self.old_logp[chain, sample] = old_logp 
                        self.new_logp[chain, sample] = new_logp
                        
                        # Accept
                        if new_logp - old_logp > unif[chain, sample]:
                            self.chains[chain, :, sample] = proposal_param
                            models[chain] = tmp_model
                        else:
                            self.chains[chain, :, sample] = self.chains[chain, :, sample - 1]


            else:
                for chain in np.arange(n_chains):
                    old_logp = models[chain].get_logp()
                    
                    chain_m = selects[chain, sample, 0]
                    chain_n = selects[chain, sample, 1]
                    
                    proposal_param = models[chain].get_param_vector() + gamma * (self.chains[chain_m, :, sample-1] - self.chains[chain_n, :, sample-1]) + bs[chain, :, sample]
                    
                    tmp_model = models[chain].copy()
                    tmp_model.set_param_vector(proposal_param)
                    
                    new_logp = tmp_model.get_logp()
                    
                    # Accept
                    if new_logp - old_logp > unif[chain, sample]:
                        self.chains[chain, :, sample] = proposal_param
                        models[chain] = tmp_model
                    else:
                        self.chains[chain, :, sample] = self.chains[chain, :, sample - 1]
                    
            pbar.animate(sample + 1)


    def get_trace(self, key):
        idx = self.model.stochastics.keys().index(key)
        return self.chains[:, idx, :]

    def plot_traces(self, burnin=0, thin=1):
        keys = sorted(self.model.stochastics.keys())
        
        for key in keys:
            plt.figure()
            plt.title(key)
            sns.distplot(self.get_trace(key)[:, burnin::thin].ravel())
            



class Node:

    def __init__(self, name, parents, value=None, observed=False):

        self.name = name

        self.observed = observed

        self.parents = {}
        
        for key, v in parents.items():
            # parameter is linked to parent node
            if hasattr(v, 'logp'):
                self.parents[key] = v
            # fixed v for parameter
            else:
                self.parents[key] = FixedValueNode('%s.%s' % (name, key), v)
                
        # Add this node as a child to all the parent nodes
        for key in self.parents.keys():
            self.parents[key].children.append(self)
                
        if value != None:
            self.value = value
        else:
            self.value = self.random()

        self.logp = None

 
    def get_logp(self):
        
        # logp is not yet calculated or is reset
        # otherwise, return cached value
        if not self.logp:
            self.logp = 0
            
            # Prior node
            for parent in self.parents.values():
                self.logp += parent.get_logp()

            self.logp += self._likelihood(self.value)
                        
            
        return self.logp
    
    def set_value(self, value):
        for child in self.children:
            child.logp = None

        self.logp = None
        self.value = value
        
    def get_value(self):
        return self.value

    def get_starting_values(self):
        self.set_value(self.random())


class FixedValueNode(Node):

    def __init__(self, name, value):
        self.children = []
        self.is_stochastic = False
        Node.__init__(self, name, {}, value=value)

    def _likelihood(self, values):
        return 0

    def random(self):
        return self.value


class Normal(Node):
    
    def __init__(self, name, mu, sigma, value=None, observed=False):
        self.is_group_parameter = True
        self.children = []

        self.is_stochastic = not observed
        
        # No parents, fixed values
        Node.__init__(self, name, {'mu':mu, 'sigma':sigma}, value, observed=observed )
    
    def _likelihood(self, values):
        return np.sum(np.log(protect_zeroes(stats.norm.pdf(loc=self.parents['mu'].get_value(), scale=self.parents['sigma'].get_value(), x=values))))

    def random(self):
        return stats.norm.rvs(loc=self.parents['mu'].get_value(), scale=self.parents['sigma'].get_value())
    
    
class TruncatedNormal(Node):
    
    def __init__(self, name, mu, sigma, lower=0, upper=np.inf, observed=False, value=None):
        
        a, b = (lower - mu) / sigma, (upper - mu) / sigma                    
        self.is_group_parameter = True
        self.children = []

        self.is_stochastic = not observed

        Node.__init__(self, name, {'mu':mu, 'sigma':sigma, 'a':a, 'b':b}, value=value )
        
    
    def _likelihood(self, values):
        return np.sum(np.log(protect_zeroes(stats.truncnorm.pdf(loc=self.parents['mu'].get_value(),
                                                                scale=self.parents['sigma'].get_value(), 
                                                                a=self.parents['a'].get_value(), 
                                                                b=self.parents['b'].get_value(), 
                                                                x=values))))
    def random(self):
        return stats.truncnorm.rvs(loc=self.parents['mu'].get_value(),
                                   scale=self.parents['sigma'].get_value(), 
                                   a=self.parents['a'].get_value(), 
                                   b=self.parents['b'].get_value(),)
    
