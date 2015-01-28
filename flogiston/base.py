import scipy as sp
from scipy import stats
import numpy as np
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
#from pyrace.crace import lba_cdf, lba_pdf, pnormP, dnormP


def pnormP(x, mean, sd):
    return np.where(np.abs(x-mean)<7.*sd, stats.norm.cdf(x, loc=mean,scale=sd), np.where(x<mean,0,1))

def dnormP(x, mean, sd):
    return np.where(np.abs(x-mean)<7.*sd,stats.norm.pdf(x, loc=mean, scale=sd),0) 

def simulate_bold(frametimes, onsets, heights, peak_delays=None):
    
    if peak_delays == None:
        peak_delays = 6

    peak_disp =1

    bold =stats.gamma.pdf(frametimes[:,np.newaxis],
                         peak_delays / peak_disp,
                         loc=onsets[np.newaxis, :],
                         scale = peak_disp)
    
    bold *= heights
    
    bold = bold.sum(1)
    
    return bold
    
    
def get_likelihood_hrf(y, frametimes, onsets, heights, peak_delays=None):
    
    y_ = simulate_bold(frametimes, onsets, heights, peak_delays)
    
    resid = y - y_

    return pm.normal_like(resid, 0, 1/np.std(resid)**2)



def sample_single_later_hrf(ter=0.1, 
                          b=1.0,
                          v=2.0, 
                          sv=1.0, 
                          onsets=np.arange(0, 1500, 20),
                          TR=2.0,
                          hrf_height_intercept=1.0, 
                          hrf_delay_intercept=6, 
                          theta_height=1.0, 
                          theta_delay=1.0,
                          upper_limit_rt=4.0):
    

    
    n_trials = len(onsets)

    nnot = n_trials
    
    rts = []
    
    vs = np.zeros((0,) )

    while nnot > 0:
        tmp_vs=stats.truncnorm.rvs((0-v)/float(sv), np.infty, loc=v, scale=sv, size=nnot)
        tmp = ter + (b / tmp_vs)
        vs = np.concatenate((vs, tmp_vs[tmp < upper_limit_rt]))

        tmp = tmp[tmp < upper_limit_rt]
        rts += list(tmp)
        nnot = n_trials - len(rts)

    rts = np.array(rts)
        
        
    # HRF  
    peak_heights = hrf_height_intercept + theta_height * (vs - v)
    peak_delays = hrf_delay_intercept + theta_delay * (vs - v)

    
    frametimes = np.arange(0, np.max(onsets) + 20, TR)

    bold = simulate_bold(frametimes, onsets, peak_heights, peak_delays)
    
    
    return rts, bold, frametimes, onsets



def sample_single_lba_hrf(ter=0.1, 
                          A=0.2,
                          b=1.0,
                          v=2.0, 
                          sv=1.0, 
                          onsets=np.arange(0, 1500, 20),
                          TR=2.0,
                          hrf_height_intercept=1.0, 
                          hrf_delay_intercept=6, 
                          theta_height=1.0, 
                          theta_delay=1.0,
                          upper_limit_rt=4.0):
    

    
    n_trials = len(onsets)

    nnot = n_trials
    
    rts = []
    
    vs = np.zeros((0,) )
    zs = np.zeros((0,) )

    while nnot > 0:
        tmp_vs=stats.truncnorm.rvs((0-v)/float(sv), np.infty, loc=v, scale=sv, size=nnot)
        tmp_zs = stats.uniform.rvs(0, A, size=nnot)
        tmp = ter + ((b - tmp_zs) / tmp_vs)

        vs = np.concatenate((vs, tmp_vs[tmp < upper_limit_rt]))
        zs = np.concatenate((zs, tmp_zs[tmp < upper_limit_rt]))

        tmp = tmp[tmp < upper_limit_rt]
        rts += list(tmp)
        nnot = n_trials - len(rts)

    rts = np.array(rts)
        
        
    # HRF  
    peak_heights = hrf_height_intercept + theta_height * (zs - (b - A/2))
    peak_delays = hrf_delay_intercept + theta_delay * (zs - (b - A/2))

    
    frametimes = np.arange(0, np.max(onsets) + 20, TR)

    bold = simulate_bold(frametimes, onsets, peak_heights, peak_delays)
    
    
    return rts, bold, frametimes, onsets
#def predict_bold(t,
                 #onsets,
                 #peak_height=1,
                 #peak_delay=6,
                 #under_delay=10,
                 #peak_disp=1,
                 #under_disp=1,
                 #p_u_ratio = 6,
                 #normalize=True,):
    
    
    #peak = peak_height * sps.gamma.pdf(t,
                     #peak_delay / peak_disp,
                     #loc=np.array([onsets]).T,
                     #scale = peak_disp)


    #under_delay = peak_delay + under_delay
    
    #undershoot = sps.gamma.pdf(t,
                           #under_delay / under_disp,
                           #loc=np.array([onsets]).T,
                           #scale = under_disp)
    
    #r = peak + peak_height / -p_u_ratio * undershoot
    
    #return r.sum(0)


#def simulate_data(conditions, frametimes):
    
    #results = np.zeros_like(frametimes)
    
    #for condition in conditions:
        #results = results + predict_bold(frametimes, **condition)
        
    #return results
