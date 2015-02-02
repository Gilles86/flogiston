from scipy import stats
from utils import dnormP, pnormP
import numpy as np


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

def generate_lba_data(ters, As, vs, svs, bs, truncated=False,  n=500):

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

    #print zs, zv, T.shape

    responses = np.argmin(T, 1)
    RTs = T.min(1)
    responses += 1

    return responses, RTs


def likelihood(responses, RTs, ters, As, vs, svs, bs):
    n_responses = len(ters)

    ll = 0

    for resp_idx in np.unique(responses):
        resp = responses[responses == resp_idx] 
        RT = RTs[responses == resp_idx] 

        for r in np.arange(0, n_responses):
            if (r+1) == resp_idx:
                #print pdf(ters[r], As[r], vs[r], svs[r], bs[r], RT)
                ll += np.sum(np.log(pdf(ters[r], As[r], vs[r], svs[r], bs[r], RT)))
            else:
                ll += np.sum(np.log(1 - cdf(ters[r], As[r], vs[r], svs[r], bs[r], RT)))


    return ll


def plot_likelihood(ters, As, vs, svs, bs, t_max=2.0, n_ts=100, color_palette='Set1', **kwargs):
    import seaborn as sns
    import matplotlib.pyplot as plt
    colors = sns.color_palette(color_palette)

    responses = np.arange(1, len(ters)+1)
    t = np.linspace(0, t_max, n_ts)

    for resp, color in zip([1, 2], colors):
        plt.plot(t, [np.exp(likelihood(np.array([resp]), np.array([tmp]), ters, As, vs, svs, bs)) for tmp in t], color=color, **kwargs)
