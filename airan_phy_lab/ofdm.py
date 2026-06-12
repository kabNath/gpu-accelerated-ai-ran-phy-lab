import numpy as np
from .config import OFDMConfig
from .modulation import qpsk_modulate, qpsk_demodulate

def build_resource_grid(data_symbols, cfg: OFDMConfig):
    grid=np.zeros(cfg.n_subcarriers,dtype=np.complex64)
    grid[cfg.pilot_indices]=cfg.pilot_value
    if len(data_symbols)>len(cfg.data_indices): raise ValueError('too many data symbols')
    grid[cfg.data_indices[:len(data_symbols)]]=data_symbols
    return grid

def extract_data_symbols(grid,cfg,n_data_symbols=None):
    data=np.asarray(grid)[cfg.data_indices]
    return data if n_data_symbols is None else data[:n_data_symbols]

def ofdm_ifft(grid): return np.fft.ifft(np.asarray(grid),norm='ortho').astype(np.complex64)
def ofdm_fft(time_signal): return np.fft.fft(np.asarray(time_signal),norm='ortho').astype(np.complex64)
def add_cp(x,cfg): return np.concatenate([x[-cfg.cp_len:],x]).astype(np.complex64)
def remove_cp(x_cp,cfg): return np.asarray(x_cp)[cfg.cp_len:cfg.cp_len+cfg.n_subcarriers]

def ofdm_transmit(bits,cfg:OFDMConfig):
    syms=qpsk_modulate(bits)[:len(cfg.data_indices)]
    grid=build_resource_grid(syms,cfg)
    return add_cp(ofdm_ifft(grid),cfg), grid, len(syms)

def equalize(grid_rx,h_est,eps=1e-8): return grid_rx/(h_est+eps)

def ofdm_receive(rx_cp,h_est,cfg,n_data_symbols):
    y=ofdm_fft(remove_cp(rx_cp,cfg)); eq=equalize(y,h_est)
    return qpsk_demodulate(extract_data_symbols(eq,cfg,n_data_symbols)), y, eq
