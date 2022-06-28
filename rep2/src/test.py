import numpy as np
import gym
import pybulletgym.envs

import time
from omegaconf import OmegaConf

# from agent import PPO, UP
from envs import CartPoleEnv, PendulumEnv

def test_agent(env_class, flags):
    total_rewards = []
    len_episodes = []

    env = env_class()
    env = env.unwrapped
    if flags.env in ['cartpole', 'pendulum']:
        env.reset()
    if flags.test_gui:
        env.render(mode='human')

    for i in range(flags.test_num_episodes):    
        env.reset()
        
        done = False
        len_episode = 0
        total_reward = 0
        while not done:
            if True: #### TODO: add agent's action
                action = env.action_space.sample()
            else:
                action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if flags.test_gui:
                env.render(mode='human')
                time.sleep(0.02)
            
            len_episode += 1
            total_reward += reward

            if len_episode >= flags.test_max_episode_len:
                break
        total_rewards.append(total_reward)
        len_episodes.append(len_episode)

    print('Reward (mean/std):', np.mean(total_rewards), np.std(total_rewards))
    print('Episode length (mean/std):', np.mean(len_episodes), np.std(len_episodes))

if __name__ == "__main__":
    flags = OmegaConf.load("configs/config.yml")

    env_class = {
        "cartpole": CartPoleEnv,
        "pendulum": PendulumEnv,
        "hopper": lambda x=None: gym.make("HopperPyBulletEnv-v0", disable_env_checker=True),
        "halfcheetah": lambda x=None: gym.make("HalfCheetahPyBulletEnv-v0", disable_env_checker=True),
        "walker": lambda x=None: gym.make("Walker2DPyBulletEnv-v0", disable_env_checker=True),
    }[flags.env]

    test_agent(env_class, flags)