from dataclasses import dataclass
import numpy as np
from .config import LinkAdaptationConfig
MCS_TABLE=[
{'mcs':0,'mod':'QPSK','rate':0.12,'snr_threshold':-5.0}, {'mcs':1,'mod':'QPSK','rate':0.19,'snr_threshold':-3.0},
{'mcs':2,'mod':'QPSK','rate':0.30,'snr_threshold':-1.0}, {'mcs':3,'mod':'QPSK','rate':0.44,'snr_threshold':2.0},
{'mcs':4,'mod':'QPSK','rate':0.59,'snr_threshold':5.0}, {'mcs':5,'mod':'16QAM','rate':0.37,'snr_threshold':8.0},
{'mcs':6,'mod':'16QAM','rate':0.48,'snr_threshold':11.0}, {'mcs':7,'mod':'64QAM','rate':0.45,'snr_threshold':15.0},
{'mcs':8,'mod':'64QAM','rate':0.65,'snr_threshold':19.0}, {'mcs':9,'mod':'256QAM','rate':0.75,'snr_threshold':24.0}]
def select_mcs_from_snr(snr_db):
    selected=0
    for row in MCS_TABLE:
        if snr_db>=row['snr_threshold']: selected=row['mcs']
    return selected
def synthetic_bler(snr_db,mcs):
    th=MCS_TABLE[mcs]['snr_threshold']; return float(1/(1+np.exp(1.2*(snr_db-th))))
@dataclass
class OLLAController:
    cfg: LinkAdaptationConfig
    def __post_init__(self): self.offset_db=self.cfg.initial_offset_db
    def select_mcs(self,measured_snr_db): return select_mcs_from_snr(measured_snr_db+self.offset_db)
    def update(self,ack:bool):
        self.offset_db += self.cfg.up_step_db if ack else -self.cfg.down_step_db
        return self.offset_db
