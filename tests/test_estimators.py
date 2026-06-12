import numpy as np
from airan_phy_lab.config import OFDMConfig
from airan_phy_lab.ofdm import ofdm_transmit,remove_cp,ofdm_fft
from airan_phy_lab.channel import rayleigh_multipath,apply_channel_time,awgn
from airan_phy_lab.estimators import ls_channel_estimate,mmse_channel_estimate
def test_estimators_shapes():
    rng=np.random.default_rng(1); cfg=OFDMConfig(); bits=rng.integers(0,2,2*len(cfg.data_indices),dtype=np.int8); tx_cp,tx_grid,_=ofdm_transmit(bits,cfg)
    rx,nv=awgn(apply_channel_time(tx_cp,rayleigh_multipath(8,rng=rng)),20.0,rng=rng); y=ofdm_fft(remove_cp(rx,cfg)); hls=ls_channel_estimate(y,tx_grid,cfg); hmmse=mmse_channel_estimate(hls,nv,8)
    assert hls.shape==(cfg.n_subcarriers,); assert hmmse.shape==(cfg.n_subcarriers,); assert np.all(np.isfinite(hmmse))
