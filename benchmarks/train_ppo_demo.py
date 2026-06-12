from airan_phy_lab.ppo_link_adaptation import train_tiny_ppo_smoke_test
if __name__=='__main__':
    model=train_tiny_ppo_smoke_test(seed=0,episodes=3); print(model); print('Tiny PPO smoke test completed.')
