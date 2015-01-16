import scipy.stats as sts

def predict_bold(t,
                 onsets,
                 peak_height=1,
                 peak_delay=6,
                 under_delay=16,
                 peak_disp=1,
                 under_disp=1,
                 p_u_ratio = 6,
                 normalize=True,):
    
    
    peak = peak_height * sps.gamma.pdf(t,
                     peak_delay / peak_disp,
                     loc=np.array([onsets]).T,
                     scale = peak_disp)
    
    undershoot = sps.gamma.pdf(t,
                           under_delay / under_disp,
                           loc=np.array([onsets]).T,
                           scale = under_disp)
    
    r = peak + peak_height / -p_u_ratio * undershoot
    
    return r.sum(0)




