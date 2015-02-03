from scipy import stats
from utils import dnormP, pnormP
import numpy as np
from sampler import Node


def pdf(ter, A, v, sv, b, t):
    """LBA PDF for a single accumulator"""
    t=np.maximum(t-ter, 1e-5) # absorbed into pdf 
    if A<1e-10: # LATER solution
        return np.maximum( 0, (b/(t**2)*dnormP(b/t, mean=v,sd=sv))
                          /pnormP(v/sv) )
    zs=t*sv
    zu=t*v
    bminuszu=b-zu
    bzu=bminuszu/zs
    bzumax=(bminuszu-A)/zs
    return np.maximum(0, ((v*(pnormP(bzu)-pnormP(bzumax)) +
                    sv*(dnormP(bzumax)-dnormP(bzu)))/A)/pnormP(v/sv))

def cdf(ter, A, v, sv, b, t):
    """LBA CDF for a single accumulator"""
    t=np.maximum(t-ter, 1e-5) # absorbed into cdf         
    if A<1e-10: # LATER solution
        return np.minimum(1, np.maximum(0, (pnormP(b/t,mean=v,sd=sv))
                                        /pnormP(v/sv) ))
    zs=t*sv
    zu=t*v
    bminuszu=b-zu
    xx=bminuszu-A
    bzu=bminuszu/zs
    bzumax=xx/zs
    tmp1=zs*(dnormP(bzumax)-dnormP(bzu))
    tmp2=xx*pnormP(bzumax)-bminuszu*pnormP(bzu)
    return np.minimum(np.maximum(0,(1+(tmp1+tmp2)/A)/pnormP(v/sv)),1)

def generate_lba_data(ters, As, vs, svs, bs, truncated=True,  n=500):

    ters = np.array(ters)
    As = np.array(As)
    vs = np.array(vs)
    bs = np.array(bs)
    svs = np.array(svs)

    n_accumulators = len(ters)
    
    if truncated:
        v_distros = [stats.truncnorm(-v / sv, np.infty, loc=v, scale=sv) for v, sv in zip(vs, svs)]
    else:
        v_distros = [stats.norm(loc=v, scale=sv) for v, sv in zip(vs, svs)]

    zs = stats.uniform.rvs(0, As, (n, n_accumulators))
    zv = np.array([dist.rvs(n) for  dist in v_distros]).T

    T = ((bs - zs) / zv) + ters

    responses = np.argmin(T, 1)
    RTs = T.min(1)
    responses += 1

    return responses, RTs


def likelihood(responses, RTs, ters, As, vs, svs, Bs, n_responses=2, n_conditions=None, conditions=None, return_individual_ll=False):

    if conditions == None:
        conditions = [1] * len(responses)
        ters = [[ter] for ter in ters]
        As = [[A] for A in As]
        vs = [[v] for v in vs]
        svs = [[sv] for sv in svs]
        Bs = [[B] for B in Bs]
        n_conditions=1
    elif not n_conditions:
        n_conditions = np.max(conditions)
        
    if not n_responses:
        n_responses = np.max(responses)

    if return_individual_ll:
        ll = np.zeros_like(RTs)
    else: 
        ll = 0
    
    for resp_idx in np.arange(1, n_responses+1):
        for c in np.arange(1, n_conditions+1):
            #print responses, (responses == resp_idx) & (conditions == c)
            resp = responses[(responses == resp_idx) & (conditions == c)] 
            RT = RTs[(responses == resp_idx) & (conditions == c)] 

            for r in np.arange(0, n_responses):
                if return_individual_ll:
                    if (r+1) == resp_idx:
                        ll[(responses == resp_idx) & (conditions == c)] += np.log(pdf(ters[r][c-1], As[r][c-1], vs[r][c-1], svs[r][c-1], As[r][c-1] + Bs[r][c-1], RT))
                    else:
                        ll[(responses == resp_idx) & (conditions == c)] += np.log(1 - cdf(ters[r][c-1], As[r][c-1], vs[r][c-1], svs[r][c-1], As[r][c-1] + Bs[r][c-1], RT))
                else:
                    if (r+1) == resp_idx:
                        ll += np.sum(np.log(pdf(ters[r][c-1], As[r][c-1], vs[r][c-1], svs[r][c-1], As[r][c-1] + Bs[r][c-1], RT)))
                    else:
                        ll += np.sum(np.log(1 - cdf(ters[r][c-1], As[r][c-1], vs[r][c-1], svs[r][c-1], As[r][c-1] + Bs[r][c-1], RT)))


    return ll




def plot_likelihood(ters, As, vs, svs, bs, t_max=2.0, n_ts=100, color_palette='Set1', **kwargs):
    import seaborn as sns
    import matplotlib.pyplot as plt
    colors = sns.color_palette(color_palette)

    responses = np.arange(1, len(ters)+1)
    t = np.linspace(0, t_max, n_ts)

    for resp, color in zip([1, 2], colors):
        plt.plot(t, [np.exp(likelihood(np.array([resp]), np.array([tmp]), ters, As, vs, svs, bs)) for tmp in t], color=color, **kwargs)




class LBA(Node):

    pars = ['ter', 'B', 'A', 'v', 'sv']
    
    def __init__(self, name, ter, v, A, B, sv=1.0, n_conditions=None, n_responses=None, value=(None, None, None), **kwargs):
        " This Node expects lists of nodes as parents!!"
        " Example: [[ter_resp1_cond1, ter_resp1_cond2], [ter_resp2_cond1, ter_resp2_cond2]] "
        
        self.stochastic = False
        self.children = []

        if not n_conditions or not n_responses:
            for l in [ter, v, A, B]:
                if type(l) == list:
                    self.n_responses = len(l)
                    self.n_conditions = len(l[0])
                    break
            raise Exception("Don't know number of conditions!")
        else:
            self.n_conditions = n_conditions
            self.n_responses = n_responses
        
        if type(ter) != list:
            ter = [[ter] * self.n_conditions] * self.n_responses
        if type(v) != list:
            v = [[v] * self.n_conditions] * self.n_responses
        if type(A) != list:
            A = [[A] * self.n_conditions] * self.n_responses
        if type(B) != list:
            B = [[B] * self.n_conditions] * self.n_responses
        if type(sv) != list:
            sv = [[sv] * self.n_conditions] * self.n_responses

        parents = {}

        self.parent_node_keys = {}

        for par in LBA.pars:
            self.parent_node_keys[par] = []
            for resp_idx in np.arange(0, self.n_responses):
                self.parent_node_keys[par].insert(resp_idx, [])
                for cond_idx in np.arange(0, self.n_conditions):
                    key = '%s.resp_%d.condition_%d' % (par, resp_idx+1, cond_idx+1)
                    self.parent_node_keys[par][resp_idx].insert(cond_idx, key)
                    parents[key] = locals()[par][resp_idx][cond_idx]

        Node.__init__(self, name, parents, value=value, **kwargs )

    
    def _likelihood(self, values):
        parameter_values = dict([(par, [[self.parents[t].get_value() for t in sl] for sl in self.parent_node_keys[par]]) for par in LBA.pars])
        #print parameter_values
        return likelihood(values[0], 
                          values[1],
                          parameter_values['ter'],
                          parameter_values['A'],
                          parameter_values['v'],
                          parameter_values['sv'],
                          parameter_values['B'],
                          n_responses=self.n_responses,
                          n_conditions=self.n_conditions,
                          conditions=values[2])

#responses, RTs, ters, As, vs, svs, bs, n_responses=2, n_conditions=None, conditions=None, return_individual_ll=False
