import os
import datetime

import numpy as np
import torch
import gym

from omegaconf import OmegaConf

from agent import PPO, UP
from envs import CartPoleEnv
from utils.buffer import ReplayMemory
from utils.logger import PlatformLogger

def eval(env_class, agent, domain_randomization_dict, flags, logger):
    env = env_class()

    domain_names = list(domain_randomization_dict.keys())
    total_rewards = [0] * flags.test_num_episodes
    for i in range(flags.test_num_episodes):
        done = False
        randomize(env, domain_randomization_dict)
        domain = get_domain(env, domain_names)
        state = env.reset()
        for _ in range(flags.test_max_episode_len):
            if flags.model_type == 'UP':
                action, _ = agent.select_action(torch.from_numpy(state), torch.from_numpy(domain))
            else:    
                action, _ = agent.select_action(torch.from_numpy(state))
            if flags.has_continuous_action_space:
                state, reward, done, info = env.step(action.detach().cpu().numpy().flatten())
            else:
                state, reward, done, info = env.step(action.item())
            total_rewards[i] += reward
            
            if done:
                break
            
    info_dict = {
        "return_mean" : float(np.mean(total_rewards)),
        "return_median" : float(np.median(total_rewards)),
        "return_var" : float(np.var(total_rewards))
    }
    logger.log_eval(info_dict)
    
def randomize(env, domain_randomization_dict):
    if len(domain_randomization_dict) <= 0:
        return
    env.set_simulator_parameters({
        domain_name: np.random.choice(random_range) for domain_name, random_range in domain_randomization_dict.items()
    })

def get_domain(env, keys):
    return np.array(env.get_simulator_parameters(keys), dtype=np.float32)

def train(env_class, agent, domain_randomization_dict, flags, logger):
    if flags.n_envs > 1:
        raise Exception

    domain_names = list(domain_randomization_dict.keys())

    #### TODO: implement multiple buffers to handle multiple envs
    if flags.model_type == 'UP':
        buffer = ReplayMemory(flags.buffer_len, ('state', 'domain', 'action', 'reward',  'done', 'logprob'))
    else:
        buffer = ReplayMemory(flags.buffer_len, ('state', 'action', 'reward',  'done', 'logprob'))
    
    envs = [env_class() for _ in range(flags.n_envs)]
    for i in range(len(envs)):
        randomize(envs[i], domain_randomization_dict)
    states = [envs[i].reset() for i in range(flags.n_envs)]

    # keep data for logging
    total_rewards = [0] * flags.n_envs
    timesteps = [0] * flags.n_envs
    returns = []
    episodes_len = []
    num_samples = 0
    num_episodes = 0
    target_update_count = 0

    for training_step in range(flags.max_training_steps):
        # Run Environments
        for i in range(flags.n_envs):
            domain = get_domain(envs[i], domain_names)

            if flags.model_type == 'UP':
                action, logprob = agent.select_action(torch.from_numpy(states[i]), torch.from_numpy(domain))
            else:
                action, logprob = agent.select_action(torch.from_numpy(states[i]))
            if flags.has_continuous_action_space:
                next_state, reward, done, info = envs[i].step(action.detach().cpu().numpy().flatten())
            else:
                next_state, reward, done, info = envs[i].step(action.item())
            if flags.model_type == 'UP':
                buffer.push(torch.tensor(states[i], device=flags.device).unsqueeze(0), \
                    torch.tensor(domain, device=flags.device).unsqueeze(0), \
                    torch.tensor([action], device=flags.device), \
                    torch.tensor([reward], device=flags.device), \
                    done, \
                    torch.tensor([logprob], device=flags.device))
            else:    
                buffer.push(torch.tensor(states[i], device=flags.device).unsqueeze(0), \
                    torch.tensor([action], device=flags.device), \
                    torch.tensor([reward], device=flags.device), \
                    done, \
                    torch.tensor([logprob], device=flags.device))

            total_rewards[i] += reward
            timesteps[i] += 1

            if done or timesteps[i] >= flags.max_episode_len:
                returns.append(total_rewards[i])
                episodes_len.append(timesteps[i])
                total_rewards[i] = 0
                timesteps[i] = 0
                num_episodes += 1
                states[i] = envs[i].reset()
                randomize(envs[i], domain_randomization_dict)
            else:
                states[i] = next_state
            
        num_samples += flags.n_envs

        # Learning from samples
        if len(buffer) >= flags.batch_size:
            batch = buffer.sample(batch_size=-1)
            agent.learn(batch)
            buffer.clear()

            target_update_count += 1
            print("[INFO] target is updated")
                
            # if training_step % flags.lr_decay_freq == 0:
            #     agent.lr_scheduler.step()
            #     print("[INFO] Learning Rate is updated")
                
        if training_step % flags.eval_freq == 0:
            eval(env_class, agent, domain_randomization_dict, flags, logger)
            print("[INFO] Evaluation is done")

        if flags.has_continuous_action_space and training_step % flags.action_std_decay_freq == 0:
            agent.decay_action_std(flags.action_std_decay_rate, flags.min_action_std)
                
        # Logging
        if training_step % flags.log_freq == 0:
            info_dict = {
                "num_samples": float(num_samples),
                "target_update_count": float(target_update_count),
                "num_episodes": float(num_episodes),
                # "epsilon": float(agent.epsilon)
            }
            if len(returns) > 0:
                info_dict.update({"episode_return_mean": sum(returns) / len(returns)})
                returns = []
            if len(episodes_len) > 0:
                info_dict.update({"episodes_len_mean": sum(episodes_len) / len(episodes_len)})
                episodes_len = []
            logger.log_iteration(info_dict)
            
        if training_step % (flags.max_training_steps // flags.checkpoint_num) == 0:
            agent.save(os.path.join(logger.result_path, f"checkpoint_{training_step}.tar"))
            print("[INFO] Checkpoint is saved")
            
    print("Train is Finished!")
    agent.save(os.path.join(logger.result_path, f"checkpoint.tar"))
                
if __name__ == "__main__":
    date_now = datetime.datetime.now().__str__()
    level1 = datetime.datetime.now().strftime(format="%y-%m-%d")
    level2 = datetime.datetime.now().strftime(format="%H-%M-%S")
    RESULT_path = os.path.join("results", level1, level2)
    if not os.path.isdir(RESULT_path):
        os.makedirs(RESULT_path)

    flags = OmegaConf.load("configs/config.yml")
    OmegaConf.save(config=flags, f=os.path.join(RESULT_path, "config.yaml"))
    
    logger = PlatformLogger(RESULT_path)
    env_class = {
        "cartpole": CartPoleEnv,
        "CartPole-v1": lambda x=None : gym.make("CartPole-v1")
    }[flags.env]

    domain_randomization_dict = {}
    if hasattr(flags, 'domain_randomization'):
        for dr in flags.domain_randomization:
            domain_name = dr[0]
            random_range = np.arange(*dr[1:])
            domain_randomization_dict.update({domain_name: random_range})

    dummy_env = env_class()
    if flags.has_continuous_action_space:
        if flags.model_type == 'UP':
            agent = UP(dummy_env.observation_space.shape[0], len(domain_randomization_dict), dummy_env.action_space.shape[0], flags, logger=logger)
        else:
            agent = PPO(dummy_env.observation_space.shape[0], dummy_env.action_space.shape[0], flags, logger=logger)
    else:
        if flags.model_type == 'UP':
            agent = UP(dummy_env.observation_space.shape[0], len(domain_randomization_dict), dummy_env.action_space.n, flags, logger=logger)
        else:
            agent = PPO(dummy_env.observation_space.shape[0], dummy_env.action_space.n, flags, logger=logger)
    train(env_class, agent, domain_randomization_dict, flags, logger)