import numpy as np
try:
    import torch, torch.nn as nn, torch.optim as optim
    HAS_TORCH=True
except ImportError:
    HAS_TORCH=False
from .link_adaptation import MCS_TABLE, synthetic_bler
from .bler_table import make_bler_fn
class LinkAdaptationEnv:
    def __init__(self, mean_snr_db=10.0, episode_len=200, seed=0, bler_fn=None):
        self.mean_snr_db=mean_snr_db; self.episode_len=episode_len; self.rng=np.random.default_rng(seed); self.t=0; self.last_ack=1.0; self.last_mcs=0.0; self.bler_fn=bler_fn or make_bler_fn()
    def reset(self): self.t=0; self.last_ack=1.0; self.last_mcs=0.0; return self._obs(self.mean_snr_db)
    def _obs(self,snr): return np.array([snr/30.0,self.last_ack,self.last_mcs/9.0],dtype=np.float32)
    def step(self,action):
        snr=self.mean_snr_db+self.rng.normal(0,2.0); action=int(np.clip(action,0,len(MCS_TABLE)-1)); bler=self.bler_fn(snr,action)
        ack=self.rng.random()>bler; reward=MCS_TABLE[action]['rate'] if ack else 0.0
        self.last_ack=float(ack); self.last_mcs=float(action); self.t+=1
        return self._obs(snr), reward, self.t>=self.episode_len, {'snr':snr,'bler':bler,'ack':ack}
if HAS_TORCH:
    class ActorCritic(nn.Module):
        def __init__(self,obs_dim=3,n_actions=10):
            super().__init__(); self.shared=nn.Sequential(nn.Linear(obs_dim,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh()); self.policy=nn.Linear(64,n_actions); self.value=nn.Linear(64,1)
        def forward(self,x):
            z=self.shared(x); return self.policy(z), self.value(z).squeeze(-1)
else: ActorCritic=None
def train_tiny_ppo_smoke_test(seed=0,episodes=5):
    if not HAS_TORCH: raise ImportError('PyTorch is required for PPO demo. Install torch first.')
    torch.manual_seed(seed); env=LinkAdaptationEnv(seed=seed); model=ActorCritic(); opt=optim.Adam(model.parameters(),lr=3e-4)
    for _ in range(episodes):
        obs=env.reset(); logps=[]; rewards=[]; values=[]; done=False
        while not done:
            obs_t=torch.tensor(obs,dtype=torch.float32).unsqueeze(0); logits,value=model(obs_t); dist=torch.distributions.Categorical(logits=logits); action=dist.sample()
            obs,reward,done,_=env.step(int(action.item())); logps.append(dist.log_prob(action)); rewards.append(torch.tensor(float(reward))); values.append(value.squeeze(0))
        G=torch.tensor(0.0); returns=[]
        for r in reversed(rewards): G=r+0.99*G; returns.insert(0,G)
        returns=torch.stack(returns); values=torch.stack(values); logps=torch.stack(logps); adv=returns-values.detach()
        loss=-(logps*adv).mean()+0.5*((values-returns)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return model
