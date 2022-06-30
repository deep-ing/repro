import argparse
import time

import gym
import numpy as np
import torch

from omegaconf import OmegaConf

from agent import PPO
from agent.envs import make_vec_envs
from agent.utils import get_vec_normalize

from train import randomize, get_domain

def test_agent(env_name, agent, flags):
    env = make_vec_envs(env_name, 12345, 1, None, None, flags, True)

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = agent.obs_rms

    if flags.test_gui:
        env.render(mode='human')
        
    domain_names = list(domain_randomization_dict.keys())
    episode_rewards = [0] * flags.test_num_episodes
    episode_len = [0] * flags.test_num_episodes
    for i in range(flags.test_num_episodes):    
        done = False
        randomize(env, domain_randomization_dict)
        domain = get_domain(env, domain_names)

        obs = env.reset()
        if agent is not None:
            eval_recurrent_hidden_states = torch.zeros(1, agent.actor_critic.recurrent_hidden_state_size, device=flags.device)
            eval_masks = torch.zeros(1, 1, device=flags.device)
        for _ in range(flags.test_max_episode_len):
            if agent is None:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = agent.actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=True)
            
            obs, reward, done, infos = env.step(action)

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=flags.device)

            if flags.test_gui:
                time.sleep(1/60)
                env.render(mode='human')

            episode_rewards[i] += reward.item()
            episode_len[i] += 1
            
            if done:
                break

    print('Reward (mean/std):', np.mean(episode_rewards), np.std(episode_rewards))
    print('Episode length (mean/std):', np.mean(episode_len), np.std(episode_len))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True)

    args = parser.parse_args()

    config_path = args.checkpoint_path.replace(args.checkpoint_path.split('/')[-1], 'config.yaml')
    
    flags = OmegaConf.load(config_path)

    domain_randomization_dict = {}
    if hasattr(flags, 'domain_randomization'):
        for dr in flags.domain_randomization:
            domain_name = dr[0]
            random_range = np.arange(*dr[1:])
            domain_randomization_dict.update({domain_name: random_range})

    dummy_env = gym.make(flags.env_name)
    domain_dim = len(domain_randomization_dict)
    agent = PPO(dummy_env.observation_space, dummy_env.action_space, domain_dim, flags)
    dummy_env.close()

    try:
        agent.load(args.checkpoint_path)
        print('\ntrained model loaded!\n')
    except:
        agent = None
        print('\ntrained model not found!\n')
    
    test_agent(flags.env_name, agent, flags)