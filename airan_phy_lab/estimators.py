import numpy as np
from .config import OFDMConfig

def linear_interpolate_complex(pilot_indices,pilot_values,n):
    x=np.arange(n); pilot_indices=np.asarray(pilot_indices); pilot_values=np.asarray(pilot_values)
    return (np.interp(x,pilot_indices,pilot_values.real)+1j*np.interp(x,pilot_indices,pilot_values.imag)).astype(np.complex64)

def ls_channel_estimate(y_grid,x_grid,cfg:OFDMConfig):
    pilots=cfg.pilot_indices; hp=y_grid[pilots]/(x_grid[pilots]+1e-8)
    return linear_interpolate_complex(pilots,hp,cfg.n_subcarriers)

def frequency_correlation_matrix(n_subcarriers,n_taps,decay=0.5):
    pdp=np.exp(-decay*np.arange(n_taps)); pdp=pdp/pdp.sum(); k=np.arange(n_subcarriers)
    R=np.zeros((n_subcarriers,n_subcarriers),dtype=np.complex64)
    for l,p in enumerate(pdp): R += p*np.exp(-1j*2*np.pi*np.subtract.outer(k,k)*l/n_subcarriers)
    return R.astype(np.complex64)

def mmse_channel_estimate(h_ls,noise_variance,n_taps=8):
    n=len(h_ls); R=frequency_correlation_matrix(n,n_taps); A=R+noise_variance*np.eye(n,dtype=np.complex64)
    Wt=np.linalg.solve(A.T,R.T); return (Wt.T @ h_ls).astype(np.complex64)

def mse(a,b): return float(np.mean(np.abs(np.asarray(a)-np.asarray(b))**2))
