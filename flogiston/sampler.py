import numpy as np
from scipy import stats
from utils import protect_zeroes, shuffle
from progressbar import ProgressBar
from collections import OrderedDict



from copy import deepcopy

class Model(object):
    
    def __init__(self, nodes):
        self.nodes = {}        
        self.stochastics = []
        
        for node in nodes:
            self.nodes[node.name] = node
            
            for key, parent in node.parents.items():
                self.stochastics.append((node, key))
        
    def get_node(self, name):
        return self.nodes[name]
    
    def copy(self):
        return deepcopy(self)
    
    def get_starting_values(self):
        for node in self.nodes.values():
            node.get_starting_values()
            
    def get_param_vector(self):
        return np.array([node.get_value(param) for node, param in self.stochastics])
    
    def set_param_vector(self, vector):
        
        for i, (node, param) in enumerate(self.stochastics):
            node.set_value(param, vector[i])
            
    def get_logp(self):
        logp = 0
        for node, param in self.stochastics:
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
        
        for sample in np.arange(1, n_samples):
            pbar.animate(sample)
            for chain in np.arange(n_chains):
                old_logp = models[chain].get_logp()
                
                chain_m = selects[chain, sample, 0]
                chain_n = selects[chain, sample, 1]
                
                proposal_param = models[chain].get_param_vector() + gamma * (self.chains[chain_m, :, sample-1] - self.chains[chain_n, :, sample-1]) + bs[chain, :, sample]
                
                models[chain].set_param_vector(proposal_param)
                
                new_logp = models[chain].get_logp()
                
                # Accept
                if new_logp - old_logp > unif[chain, sample]:
                    self.chains[chain, :, sample] = proposal_param
                else:
                    self.chains[chain, :, sample] = self.chains[chain, :, sample - 1]
                    models[chain].set_param_vector(self.chains[chain, :, sample - 1])
                    models[chain].logp = old_logp


        def plot_posterior(self, parameter):
            



class Node:

    def __init__(self, name, parameters, value=None):

        self.name = name

        self.parents = {}
        self.parameters = {}

        for key, v in parameters.items():
            # {'parameter':(parent, 0.2) syntax
            if type(v) == tuple:
                self.parents[key] = v[0]
                self.parameters[key] = v[1]
            # parameter is linked to parent node
            elif hasattr(v, 'logp'):
                self.parents[key] = v
                self.parameters[key] = None
            # fixed v for parameter
            else:
                self.parameters[key] = v
                
        # Add this node as a child to all the parent nodes
        for key in self.parents.keys():
            self.parents[key].children.append((self, key))
                
        # Set parameter values for non-initialized ones
        for key, v in self.parameters.items():
            if v == None:
                self.parameters[key] = self.parents[key].random()
        
        # If observed, set value
        if value != None:
            self.value = value
            self.observed = True
        else:
            self.observed = False
        
        #print 'parameters: %s' % str(self.parameters)
        #print 'parents: %s' % str(self.parents)

        self.logp = None
        #self.get_logp()

 
    def get_logp(self):
        
        # logp is not yet calculated or is reset
        # otherwise, return cached value
        if not self.logp:
            self.logp = 0
            
            # Prior node
            if not self.observed:            
                x = [c.get_value(value) for c, value in self.children]
                self.logp += self._likelihood(x)
            # Likelihood node
            else:
                for key, parent in self.parents.items():
                    self.logp += parent._likelihood(self.parameters[key])

                self.logp += self._likelihood(self.value)
                        
            
        return self.logp
    
    def set_value(self, parameter, value):
        self.parameters[parameter] = value
        self.logp = None
        
        if not self.is_group_parameter:
            for parent in self.parents:
                parents.logp = None
        
    def get_value(self, parameter):
        return self.parameters[parameter]

    def get_starting_values(self):
        for key, parent in self.parents.items():
            self.parameters[key] = parent.random()


class FixedValueNode(Node):

    def __init__(self, name, value):
        self.children = []
        Node.__init__(self, name, {}, {name:value})

    def _likelihood(self, values):
        return 0

    def random(self):
        return self.parameters[self.name]


class Normal(Node):
    
    def __init__(self, name, mu, sigma, value=None):
        self.is_group_parameter = True
        self.children = []
        
        # No parents, fixed values
        Node.__init__(self, name, {'mu':mu, 'sigma':sigma}, value )
    
    def _likelihood(self, values):
        return np.sum(np.log(protect_zeroes(stats.norm.pdf(loc=self.parameters['mu'], scale=self.parameters['sigma'], x=values))))

    def random(self):
        return stats.norm.rvs(loc=self.parameters['mu'], scale=self.parameters['sigma'])
    
    
class TruncatedNormal(Node):
    
    def __init__(self, name, mu, sigma, lower=0, upper=np.inf):
        
        a, b = (lower - mu) / sigma, (upper - mu) / sigma                    
        self.is_group_parameter = True
        self.children = []

        Node.__init__(self, name, {'mu':mu, 'sigma':sigma, 'a':a, 'b':b}, )
        
    
    def _likelihood(self, values):
        return np.sum(np.log(protect_zeroes(stats.truncnorm.pdf(loc=self.parameters['mu'],
                                                                scale=self.parameters['sigma'], 
                                                                a=self.parameters['a'], 
                                                                b=self.parameters['b'], 
                                                                x=values))))
    def random(self):
        return stats.truncnorm.rvs(loc=self.parameters['mu'],
                                   scale=self.parameters['sigma'], 
                                   a=self.parameters['a'], 
                                   b=self.parameters['b'],)
    
