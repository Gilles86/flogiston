import numpy as np
from scipy import stats
from utils import protect_zeroes


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
        self.get_logp()

 
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
        return self.value[parameter]


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
    
