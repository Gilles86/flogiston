from scipy import stats
from utils import dnormP, pnormP
import numpy as np
from sampler import Node
from bold import simulate_bold, get_likelihood_hrf
from utils import pack_conditions_responses_param

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

def generate_lba_data(ters, As, vs, svs, bs, bold=None, height_intercept=1.0, delay_intercept=6.0, height_theta=1.0, delay_theta=0.0, truncated=True, trial_length=10, TR=2.0, n=500):

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

    if bold is not None:
        expression = 0
        if bold == 'first_drift':
            expression = zv[:, 0] - vs[0]
        if bold == 'winning_drift':
            print 'winning drift'
            expression = zv[np.arange(zv.shape[0]), responses-1] - vs[responses-1]
        elif bold == 'total_drift':
            expression = zv.sum(1) - vs.sum()
        elif bold == 'winning_distance':
            expression = -zs[np.arange(zs.shape[0]),responses-1] - As[responses-1]/2 #bs[0] - zs[:,0] - bs[0] - A/2
            print 'winning distance'
        elif bold == 'first_distance':
            expression = zs[:,0]#bs[0] - zs[:,0] - bs[0] - A/2
        elif bold == 'total_distance':
            expression = (zv * RTs[:, np.newaxis]).sum(1) - (zv * RTs[:, np.newaxis]).sum(1).mean()
        
        heights = height_intercept + (expression) * height_theta
        delays = delay_intercept + (expression) * delay_theta

        onsets = np.arange(0, n*trial_length,trial_length)
        frametimes = np.arange(0,(n+1)*trial_length,TR)

        bold = simulate_bold(frametimes, onsets, heights, peak_delays=delays)

        return responses, RTs, bold, frametimes, onsets

    return responses, RT


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

    for resp, color in zip(responses, colors):
        plt.plot(t, np.exp(likelihood(np.array([resp] * len(t)), t, ters, As, vs, svs, bs, return_individual_ll=True)), color=color, **kwargs)


def bold_likelihood(responses, RTs, conditions, bold, frametimes, onsets, ter=None, v=None, A=None, B=None, kind='winning_drift', n_integration_points=10, sv=None, height_intercept=1.0, delay_intercept=6.0, height_theta=1.0, delay_theta=0.0, n_conditions=1, n_responses=2):

    if sv == None:
        sv = np.ones_like(ter)

    expression = 0

    if kind == 'winning_drift':

        if np.all(A == 0):
            T = RTs - ter[responses - 1, conditions - 1]
            emp_v = B[responses-1, conditions-1] / T
            #expression = emp_v - v[responses-1, conditions-1]
            expression = emp_v - emp_v.mean()

    if kind == 'winning_distance':
        #print 'nigger'

        quantiles = np.linspace(0, 1, n_integration_points, endpoint=False) + (1. / n_integration_points / 2.)
        n = stats.norm(loc=v[..., np.newaxis, ], scale=sv[..., np.newaxis])
        possible_vs = n.ppf(quantiles)

        pdfs = stats.norm().pdf(stats.norm().ppf(quantiles))

        likelihood = 0

        T = RTs - ter[responses - 1, conditions - 1]

        for quantile in np.arange(n_integration_points):
            current_distances = T * possible_vs[responses - 1, conditions - 1, quantile]
            expression = current_distances
            #expression -= B[responses - 1, conditions - 1] + (A[responses - 1, conditions - 1]/2.)
            expression -= T * v[responses - 1, conditions - 1]

            heights = height_intercept[responses - 1, conditions -1] + expression * height_theta[responses - 1, conditions -1]
            delays = delay_intercept[responses - 1, conditions -1] + expression * delay_theta[responses - 1, conditions -1]
            
            ll_bold = np.exp(np.array(get_likelihood_hrf(bold, frametimes, onsets, heights, dispersions=1, peak_delays=delays)).astype(np.longdouble))
            
            likelihood += pdfs[quantile] * 1. / n_integration_points * ll_bold
            
        likelihood = np.log(likelihood)
        return likelihood
    
    heights = height_intercept[responses - 1, conditions -1] + expression * height_theta[responses - 1, conditions -1]
    delays = delay_intercept[responses - 1, conditions -1] + expression * delay_theta[responses - 1, conditions -1]

    return get_likelihood_hrf(bold, frametimes, onsets, heights, dispersions=1, peak_delays=delays)


class LBA(Node):

    pars = ['ter', 'B', 'A', 'v', 'sv']
    
    def __init__(self, name, ter, v, A, B, sv=1.0, n_conditions=None, n_responses=None, value=(None, None, None), **kwargs):
        " This Node expects lists of nodes as parents!!"
        " Example: [[ter_resp1_cond1, ter_resp1_cond2], [ter_resp2_cond1, ter_resp2_cond2]] "
        
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
       
        ter = pack_conditions_responses_param(ter, self.n_conditions, self.n_responses)
        v = pack_conditions_responses_param(v, self.n_conditions, self.n_responses)
        A = pack_conditions_responses_param(A, self.n_conditions, self.n_responses)
        B = pack_conditions_responses_param(B, self.n_conditions, self.n_responses)
        sv = pack_conditions_responses_param(sv, self.n_conditions, self.n_responses)
        
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

        Node.__init__(self, name, parents, value=value, observed=True, **kwargs )

    
    def _likelihood(self, values):
        parameter_values = dict([(par, [[self.parents[t].get_value() for t in sl] for sl in self.parent_node_keys[par]]) for par in LBA.pars])
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

class LBA_BOLD(Node):

    pars = ['ter', 'B', 'A', 'v', 'sv', 'height_intercept', 'height_theta', 'delay_intercept', 'delay_theta']
    
    def __init__(self, name, ter, v, A, B, height_intercept=1.0, height_theta=0.0, delay_intercept=6.0, delay_theta=0.0, sv=1.0, n_conditions=None, n_responses=None, value=(None, None, None, None, None, None), kind='winning_drift', **kwargs):
        " This Node expects lists of nodes as parents!!"
        " Example: [[ter_resp1_cond1, ter_resp1_cond2], [ter_resp2_cond1, ter_resp2_cond2]] "
        " Value (responses, RT, conditions, bold, frametimes, onsets) "
        
        self.stochastic = False
        self.children = []

        self.kind = kind

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
        
        ter = pack_conditions_responses_param(ter, self.n_conditions, self.n_responses)
        v = pack_conditions_responses_param(v, self.n_conditions, self.n_responses)
        A = pack_conditions_responses_param(A, self.n_conditions, self.n_responses)
        B = pack_conditions_responses_param(B, self.n_conditions, self.n_responses)
        sv = pack_conditions_responses_param(sv, self.n_conditions, self.n_responses)
        height_intercept = pack_conditions_responses_param(height_intercept, self.n_conditions, self.n_responses)
        height_theta = pack_conditions_responses_param(height_theta, self.n_conditions, self.n_responses)
        delay_intercept = pack_conditions_responses_param(delay_intercept, self.n_conditions, self.n_responses)
        delay_theta = pack_conditions_responses_param(delay_theta, self.n_conditions, self.n_responses)

        parents = {}

        self.parent_node_keys = {}

        for par in LBA_BOLD.pars:
            self.parent_node_keys[par] = []
            for resp_idx in np.arange(0, self.n_responses):
                self.parent_node_keys[par].insert(resp_idx, [])
                for cond_idx in np.arange(0, self.n_conditions):
                    key = '%s.resp_%d.condition_%d' % (par, resp_idx+1, cond_idx+1)
                    self.parent_node_keys[par][resp_idx].insert(cond_idx, key)
                    parents[key] = locals()[par][resp_idx][cond_idx]

        Node.__init__(self, name, parents, value=value, **kwargs )

    
    def _likelihood(self, values):
        parameter_values = dict([(par, np.array([[self.parents[t].get_value() for t in sl] for sl in self.parent_node_keys[par]])) for par in LBA_BOLD.pars])
        return bold_likelihood(values[0], 
                          values[1],
                          values[2],
                          values[3],
                          values[4],
                          values[5],
                          n_responses=self.n_responses,
                          n_conditions=self.n_conditions,
                          kind=self.kind,
                          **parameter_values)


#def bold_likelihood(responses, RTs, conditions, bold, frametimes, onsets, ter=None, v=None, A=None, B=None, kind='winning_drift', sv=None, height_intercept=1.0, delay_intercept=6.0, height_theta=1.0, delay_theta=0.0, n_conditions=1, n_responses=2):
 
