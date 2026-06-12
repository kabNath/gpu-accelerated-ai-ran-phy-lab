import numpy as np
from airan_phy_lab.config import OFDMConfig
from airan_phy_lab.ofdm import ofdm_transmit,ofdm_receive
from airan_phy_lab.modulation import hard_decision_ber
def test_ofdm_no_channel_roundtrip():
    rng=np.random.default_rng(0); cfg=OFDMConfig(); bits=rng.integers(0,2,2*len(cfg.data_indices),dtype=np.int8); tx_cp,_,n=ofdm_transmit(bits,cfg)
    out,_,_=ofdm_receive(tx_cp,np.ones(cfg.n_subcarriers,dtype=np.complex64),cfg,n); assert hard_decision_ber(bits,out)==0.0
