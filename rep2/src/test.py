import time

import gym
import numpy as np
import torch

from omegaconf import OmegaConf

from agent import PPO

from envs import CartPoleEnv, PendulumEnv

def randomize(env, domain_randomization_dict):
    if len(domain_randomization_dict) <= 0:
        return
    env.set_simulator_parameters({
        domain_name: np.random.choice(random_range) for domain_name, random_range in domain_randomization_dict.items()
    })

def get_domain(env, keys):
    if len(keys) <= 0:
        return np.array([])
    return np.array(env.get_simulator_parameters(keys), dtype=np.float32)

def test_agent(env_class, agent, flags):
    total_rewards = []
    len_episodes = []

    env = env_class()
    env = env.unwrapped
    domain_names = list(domain_randomization_dict.keys())

    if flags.env in ['cartpole', 'pendulum']:
        env.reset()
    if flags.test_gui:
        env.render(mode='human')

    total_rewards = [0] * flags.test_num_episodes
    len_episodes = [0] * flags.test_num_episodes
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
                    value, action, action_log_prob, recurrent_hidden_states = agent.actor_critic.act(
                        torch.from_numpy(obs).unsqueeze(0).to(flags.device),
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=True)
                    action = action.detach().cpu().numpy().flatten()
            
            obs, reward, done, infos = env.step(action)
            if flags.test_gui:
                env.render(mode='human')
                time.sleep(0.02)

            total_rewards[i] += reward
            len_episodes[i] += reward
            
            if done:
                break

    print('Reward (mean/std):', np.mean(total_rewards), np.std(total_rewards))
    print('Episode length (mean/std):', np.mean(len_episodes), np.std(len_episodes))

if __name__ == "__main__":
    flags = OmegaConf.load("configs/config.yml")

    env_class = {
        "cartpole": CartPoleEnv,
        "pendulum": PendulumEnv,
        "hopper": lambda x=None: gym.make("HopperPyBulletEnv-v0"),
        "halfcheetah": lambda x=None: gym.make("HalfCheetahPyBulletEnv-v0"),
        "walker": lambda x=None: gym.make("Walker2DPyBulletEnv-v0"),
    }[flags.env]

    domain_randomization_dict = {}
    if hasattr(flags, 'domain_randomization'):
        for dr in flags.domain_randomization:
            domain_name = dr[0]
            random_range = np.arange(*dr[1:])
            domain_randomization_dict.update({domain_name: random_range})

    domain_dim = len(domain_randomization_dict)

    dummy_env = env_class()
    agent = agent = PPO(dummy_env.observation_space, dummy_env.action_space, domain_dim, flags)

    try:
        agent.load(flags.model_path)
        print('\ntrained model loaded!\n')
    except:
        print('\ntrained model not found!\n')
        agent = None
    
    test_agent(env_class, agent, flags)