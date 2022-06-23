import os
import datetime

import numpy as np
import torch
import gym

from omegaconf import OmegaConf

from agent.PPO  import PPO
from envs import CartPoleEnv
from utils.buffer import ReplayMemory
from utils.logger import PlatformLogger

def eval(env_class, agent, flags, logger):
    env = env_class()

    total_rewards = [0] * flags.test_num_episodes
    for i in range(flags.test_num_episodes):
        done = False
        state = env.reset()
        for _ in range(flags.text_max_episode_len):
            action = agent.select_action(torch.from_numpy(state))
            state, reward, done, info = env.step(action)
            total_rewards[i] += reward
            
            if done:
                break
            
    info_dict = {
        "return_mean" : float(np.mean(total_rewards)),
        "return_median" : float(np.median(total_rewards)),
        "return_var" : float(np.var(total_rewards))
    }
    logger.log_eval(info_dict)
    
def train(env_class, agent, flags, logger):
    if flags.n_envs > 1:
        raise Exception

    #### TODO: implement multiple buffers to handle multiple envs
    buffer = ReplayMemory(flags.buffer_len, ('state', 'action', 'reward',  'done', 'logprob'))
    
    envs = [env_class() for _ in range(flags.n_envs)]
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
            action, logprob = agent.select_action(torch.from_numpy(states[i]))
            if flags.has_continuous_action_space:
                next_state, reward, done, info = envs[i].step(action.detach().cpu().numpy().flatten())
            else:
                next_state, reward, done, info = envs[i].step(action.item())
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
            else:
                states[i] = next_state
            
        num_samples += flags.n_envs

        # Learning from samples
        if len(buffer) >= flags.batch_size:
            batch = buffer.sample(flags.batch_size)
            agent.learn(batch)
            buffer.clear()

            target_update_count += 1
            print("[INFO] target is updated")
                
            # if training_step % flags.lr_decay_freq == 0:
            #     agent.lr_scheduler.step()
            #     print("[INFO] Learning Rate is updated")
                
            if training_step % flags.eval_freq == 0:
                eval(env_class, agent, flags, logger)
                print("[INFO] Evaluation is done")
                
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
    flags = OmegaConf.load("configs/config.yml")
    date_now = datetime.datetime.now().__str__()
    level1 = datetime.datetime.now().strftime(format="%y-%m-%d")
    level2 = datetime.datetime.now().strftime(format="%H-%M-%S")
    RESULT_path = os.path.join("results", level1, level2)
    if not os.path.isdir(RESULT_path):
        os.makedirs(RESULT_path)
    OmegaConf.save(config=flags, f=os.path.join(RESULT_path, "config.yaml"))
    
    logger = PlatformLogger(RESULT_path)
    env_class = {
        # "simple_maze" :SimpleMazeEnv,
        # "catcher" : CatcherEnv,
        # "maze":MazeEnv,
        "cartpole": CartPoleEnv,
        "CartPole-v1": lambda x=None : gym.make("CartPole-v1")
    }[flags.env]

    dummy_env = env_class()
    agent = PPO(dummy_env.observation_space.shape[0], dummy_env.action_space.n, flags, logger=logger)
    train(env_class, agent, flags, logger)