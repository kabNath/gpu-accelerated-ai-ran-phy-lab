from airan_phy_lab.link_adaptation import select_mcs_from_snr,OLLAController
from airan_phy_lab.config import LinkAdaptationConfig
def test_mcs_monotonic(): assert select_mcs_from_snr(-10)<=select_mcs_from_snr(0)<=select_mcs_from_snr(10)<=select_mcs_from_snr(30)
def test_olla_ack_increases_offset():
    o=OLLAController(LinkAdaptationConfig()); b=o.offset_db; assert o.update(True)>b
def test_olla_nack_decreases_offset():
    o=OLLAController(LinkAdaptationConfig()); b=o.offset_db; assert o.update(False)<b
