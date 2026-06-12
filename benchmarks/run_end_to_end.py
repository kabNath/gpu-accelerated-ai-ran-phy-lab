import numpy as np
from airan_phy_lab.config import OFDMConfig, LinkAdaptationConfig
from airan_phy_lab.channel import rayleigh_multipath, apply_channel_time, awgn, channel_frequency_response
from airan_phy_lab.ofdm import ofdm_transmit, ofdm_receive, remove_cp, ofdm_fft
from airan_phy_lab.estimators import ls_channel_estimate, mmse_channel_estimate, mse
from airan_phy_lab.modulation import hard_decision_ber
from airan_phy_lab.link_adaptation import OLLAController

def main():
    rng=np.random.default_rng(42); cfg=OFDMConfig(n_subcarriers=64,cp_len=16,pilot_spacing=4); snr_db=20.0
    tx_bits=rng.integers(0,2,2*len(cfg.data_indices),dtype=np.int8); tx_cp,tx_grid,n_data_symbols=ofdm_transmit(tx_bits,cfg)
    h=rayleigh_multipath(n_taps=8,rng=rng); rx_clean=apply_channel_time(tx_cp,h); rx_noisy,noise_var=awgn(rx_clean,snr_db,rng=rng)
    y_grid=ofdm_fft(remove_cp(rx_noisy,cfg)); h_ls=ls_channel_estimate(y_grid,tx_grid,cfg); h_mmse=mmse_channel_estimate(h_ls,noise_var,n_taps=8)
    h_true=channel_frequency_response(h,cfg.n_subcarriers); bits_ls,_,_=ofdm_receive(rx_noisy,h_ls,cfg,n_data_symbols); bits_mmse,_,_=ofdm_receive(rx_noisy,h_mmse,cfg,n_data_symbols)
    print(f'SNR={snr_db:.1f} dB'); print(f'Noise variance: {noise_var:.3e}'); print(f'Channel MSE LS  : {mse(h_true,h_ls):.3e}'); print(f'Channel MSE MMSE: {mse(h_true,h_mmse):.3e}')
    print(f'BER with LS     : {hard_decision_ber(tx_bits,bits_ls):.3e}'); print(f'BER with MMSE   : {hard_decision_ber(tx_bits,bits_mmse):.3e}')
    print(f'Selected MCS by OLLA at {snr_db:.1f} dB: {OLLAController(LinkAdaptationConfig()).select_mcs(snr_db)}')
if __name__=='__main__': main()
