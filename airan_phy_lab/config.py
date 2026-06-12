from dataclasses import dataclass

@dataclass(frozen=True)
class OFDMConfig:
    n_subcarriers: int = 64
    cp_len: int = 16
    pilot_spacing: int = 4
    pilot_value: complex = 1 + 0j
    @property
    def pilot_indices(self): return list(range(0, self.n_subcarriers, self.pilot_spacing))
    @property
    def data_indices(self):
        pilots=set(self.pilot_indices)
        return [i for i in range(self.n_subcarriers) if i not in pilots]

@dataclass
class LinkAdaptationConfig:
    target_bler: float = 0.10
    up_step_db: float = 0.1
    down_step_db: float = 0.9
    initial_offset_db: float = 0.0
