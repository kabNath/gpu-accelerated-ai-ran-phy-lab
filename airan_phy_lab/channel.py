import numpy as np

def exponential_pdp(n_taps,decay=0.5):
    taps=np.exp(-decay*np.arange(n_taps)); return taps/taps.sum()

def rayleigh_multipath(n_taps=8,rng=None):
    rng=np.random.default_rng() if rng is None else rng; pdp=exponential_pdp(n_taps)
    h=(rng.normal(size=n_taps)+1j*rng.normal(size=n_taps))/np.sqrt(2)
    return (h*np.sqrt(pdp)).astype(np.complex64)

def apply_channel_time(x,h): return np.convolve(np.asarray(x),np.asarray(h),mode='full')[:len(x)].astype(np.complex64)

def awgn(x,snr_db,rng=None):
    rng=np.random.default_rng() if rng is None else rng; x=np.asarray(x)
    p=np.mean(np.abs(x)**2); nv=p/(10**(snr_db/10))
    noise=np.sqrt(nv/2)*(rng.normal(size=x.shape)+1j*rng.normal(size=x.shape))
    return (x+noise).astype(np.complex64), float(nv)

def channel_frequency_response(h,n_subcarriers):
    hpad=np.zeros(n_subcarriers,dtype=np.complex64); hpad[:len(h)]=h
    return np.fft.fft(hpad,norm='ortho').astype(np.complex64)
