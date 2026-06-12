import numpy as np

def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    bits=np.asarray(bits).astype(np.int8).reshape(-1)
    if len(bits)%2: bits=np.pad(bits,(0,1))
    pairs=bits.reshape(-1,2)
    i=1-2*pairs[:,0]; q=1-2*pairs[:,1]
    return ((i+1j*q)/np.sqrt(2)).astype(np.complex64)

def qpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    symbols=np.asarray(symbols)
    return np.vstack([(symbols.real<0).astype(np.int8),(symbols.imag<0).astype(np.int8)]).T.reshape(-1)

def hard_decision_ber(tx_bits, rx_bits) -> float:
    n=min(len(tx_bits),len(rx_bits))
    return 0.0 if n==0 else float(np.mean(np.asarray(tx_bits[:n]) != np.asarray(rx_bits[:n])))
