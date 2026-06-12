import numpy as np
from airan_phy_lab.modulation import qpsk_modulate,qpsk_demodulate,hard_decision_ber
def test_qpsk_roundtrip():
    bits=np.array([0,0,0,1,1,1,1,0],dtype=np.int8); assert hard_decision_ber(bits,qpsk_demodulate(qpsk_modulate(bits)))==0.0
