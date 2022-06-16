from agent.CRAR  import CRAR 
from omegaconf import OmegaConf
from envs import SimpleMazeEnv, MazeEnv, CatcherEnv
from utils.buffer import RolloutBuffer
from utils.logger import PlatformLogger
import os 
import gym 
import datetime
import torch 
import numpy as np 

def eval(env_class, agent, flags, logger, obs_preprocessing):
    env = env_class()

    agent_eps = agent.epsilon
    agent.epsilon = flags.test_epsilon 
    flags.random_action = False
    rewards = [0 for i in range(flags.test_episodes)]
    for i in range(flags.test_episodes):    
        done = False 
        timestep = 0 
        state = env.reset()
        while not done:
            act = agent.act(obs_preprocessing(torch.tensor(state, device=flags.device)).unsqueeze(0))
            ns, reward, done, info = env.step(act)
            rewards[i] += reward
            if timestep >= flags.test_horizon:
                done = True
            timestep += 1
    info_dict = {
        "timestep": float(timestep),
        "return_mean" : float(np.mean(rewards)),
        "return_median" : float(np.median(rewards)),
        "return_var" : float(np.var(rewards))
    }
    logger.log_eval(info_dict)
    agent.epsilon = agent_eps
    flags.random_action = True
    
def train(env_class, agent, flags, logger):
    obs_preprocessing = lambda x : x.unsqueeze(0)
    # obs_preprocessing = lambda x : x  # CartPole
    buffer = RolloutBuffer(flags.buffer_len, ['state', 'action', 'reward',  'done', 'next_state'], obs_preprocessing)
    
    envs = [env_class() for i in range(flags.n_envs)] 
    envs_states = [env.reset() for env in envs]
    envs_dones = [False for i in range(flags.n_envs)]
    envs_reward = [0 for i in range(flags.n_envs)]
    envs_timesteps = [0 for i in range(flags.n_envs)]

    timestep = 0 
    episode_count = 0 
    returns = []
    episode_steps = []


    target_update_count = 0
    train_count = 0
    checkpoint_timestep = flags.timesteps // flags.checkpoint_num
    time_to_checkpoint = checkpoint_timestep
    while timestep < flags.timesteps:
        timestep += flags.n_envs 
        if len(buffer) >= flags.batch_size and timestep > flags.train_start_timestep:
            if train_count > flags.max_training_epoch:
                break
            train_count += 1
            for j in range(flags.learn_epoch):
                env_batch = buffer.sample(flags.batch_size, flags.device)
                agent.learn(env_batch)
            
            if timestep % flags.log_freq  == 0:
                agent.save(os.path.join(logger.result_path, f"checkpoint.tar"))
                # agent.load(os.path.join(logger.result_path, f"checkpoint.tar"))
                
                info_dict = {
                    "timestep": float(timestep),
                    "target_update_count" : float(target_update_count),
                    "episode_count" : float(episode_count),
                    "epsilon" : float(agent.epsilon)
                }
                if len(returns) > 0 :
                    info_dict.update({"episode_return_mean":sum(returns) / len(returns)})
                    returns = []
                if len(episode_steps) >0:
                    info_dict.update({"episode_steps_mean":sum(episode_steps) / len(episode_steps)})
                    episode_steps = []
                logger.log_iteration(info_dict)
                
            if timestep % flags.target_update_freq == 0:
                agent.update_target()
                target_update_count += 1
            if timestep % flags.lr_decay_freq == 0:
                agent.lr_schduler.step()
                
                target_update_count += 1
            if timestep % flags.eval_freq == 0:
                eval(env_class, agent, flags, logger, obs_preprocessing)
                
        agent.anneal_epsilon(timestep)
        if timestep > time_to_checkpoint:
            agent.save(os.path.join(logger.result_path, f"checkpoint_{time_to_checkpoint/flags.timesteps:.1f}.tar"))
            # agent.load(os.path.join(logger.result_path, f"checkpoint_{time_to_checkpoint/flags.timesteps:.1f}.tar"))
            time_to_checkpoint += checkpoint_timestep
        
        # Run Environments
        envs_states = [envs_states[i] if not envs_dones[i] else envs[i].reset() for i in range(len(envs_states))]
        actions = [agent.act(obs_preprocessing(torch.tensor(state, device=flags.device)).unsqueeze(0)) for state in envs_states]
        steps = [envs[i].step(actions[i]) for i in range(len(envs))]
        for i, (ns, r, done, info) in enumerate(steps):
            buffer.append({k:v for k,v in zip(['state', 'action', 'reward',  'done', 'next_state'], 
                                                [envs_states[i].copy(), actions[i], r, done,  ns])}, device=flags.device)
            envs_states[i] = ns
            envs_reward[i] += r
            envs_timesteps[i] += 1
            if envs_timesteps[i] > flags.horizon:
                done = True
            envs_dones[i] = done
            if done:
                episode_count +=1 
                returns.append(envs_reward[i])
                episode_steps.append(envs_timesteps[i])
                envs_reward[i] = 0
                envs_timesteps[i] = 0
                
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
                    "simple_maze" :SimpleMazeEnv,
                    "catcher" : CatcherEnv,
                    "maze":MazeEnv,
                    "CartPole-v1":lambda x=None : gym.make("CartPole-v1")
                }[flags.env]

    dummy_env = env_class()
    agent = CRAR(dummy_env.observation_space, dummy_env.action_space, flags, logger=logger)
    train(env_class, agent, flags, logger)