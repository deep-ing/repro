### NEED TO BE REIMPLEMENTED: CartPole

from agent.UP_OSI  import UP_OSI 
from envs import CartPoleEnv
from omegaconf import OmegaConf
from utils.buffer import RolloutBuffer
from utils.logger import PlatformLogger
import os 
import gym 
import datetime
import torch 
from torch import nn
import numpy as np 

# def eval(env_class, agent, mode, flags, logger, obs_preprocessing):
#     env = env_class()
#     agent_eps = agent.epsilon
#     agent.epsilon = flags.test_epsilon 
#     flags.random_action = False
#     rewards = [0 for i in range(flags.test_episodes)]
#     for i in range(flags.test_episodes):
#         done = False 
#         timestep = 0
#         state = env.reset()
#         while not done:
#             act = agent.act(obs_preprocessing(torch.tensor(state, device=flags.device)).unsqueeze(0))
#             state, reward, done, info = env.step(act)
#             rewards[i] += reward
#             if timestep >= flags.test_horizon:
#                 done = True
#             timestep += 1
#     info_dict = {
#         "timestep": float(timestep),
#         "return_mean" : float(np.mean(rewards)),
#         "return_median" : float(np.median(rewards)),
#         "return_var" : float(np.var(rewards))
#     }
#     logger.log_eval(info_dict)
#     agent.epsilon = agent_eps
#     flags.random_action = True
    
def randomize(env):
    env.set_simulator_parameters({
        'masspole': np.random.choice(np.arange(0.5, 1.51, 0.25)),
        'length': np.random.choice(np.arange(0.5, 1.51, 0.25)),
    })

def get_domains(env, keys=['masspole', 'length']):
    return env.get_simulator_parameters(keys)

def train(env_class, agent, mode, flags, logger):
    # obs_preprocessing = lambda x : x.unsqueeze(0)
    obs_preprocessing = lambda x : x  # CartPole
    buffer = RolloutBuffer(flags.buffer_len, ['states', 'actions', 'rewards',  'dones', 'next_states', 'domains'], obs_preprocessing)
    
    envs = [env_class() for i in range(flags.n_envs)] 
    for i in range(len(envs)):
        randomize(envs[i])
    envs_states = [env.reset() for env in envs]
    envs_reward = [0 for i in range(flags.n_envs)]
    envs_timesteps = [0 for i in range(flags.n_envs)]
    envs_history = [{} for i in range(flags.n_envs)]
    for key in ['states', 'actions', 'rewards', 'dones', 'next_states', 'domains']:
        for i in range(flags.n_envs):
            envs_history[i][key] = []

    timestep = 0 
    num_samples = 0
    episode_count = 0 
    returns = []
    episode_steps = []

    target_update_count = 0
    checkpoint_timestep = flags.timesteps // flags.checkpoint_num
    time_to_checkpoint = checkpoint_timestep
    while timestep < flags.timesteps:
        num_samples += flags.n_envs 
        timestep += 1

        if timestep > flags.train_start_timestep:
            if mode == 0:
                env_batch = buffer.sample(flags.batch_size, 1, flags.device)
            else:
                env_batch = buffer.sample(flags.batch_size, flags.len_history, flags.device)
            
            agent.learn(env_batch, mode=mode)
                
            if timestep % flags.target_update_freq == 0:
                agent.update_target()
                target_update_count += 1
                print("[INFO] target is updated")
            if timestep % flags.lr_decay_freq == 0:
                agent.lr_schduler.step()
                print("[INFO] Learning Rate is updated")
                
            if timestep % flags.eval_freq == 0:
                # eval(env_class, agent, mode, flags, logger, obs_preprocessing)
                print("[INFO] Evaludation is done")
                
        if timestep % flags.log_freq == 0:
            info_dict = {
                "num_samples": float(num_samples),
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
        
        if mode == 0:
            agent.anneal_epsilon(timestep)
        if timestep > time_to_checkpoint:
            agent.save(os.path.join(logger.result_path, f"mode{mode}_checkpoint_{time_to_checkpoint/flags.timesteps:.1f}.tar"))
            time_to_checkpoint += checkpoint_timestep
            print("[INFO] Checkpoint is saved")
            
        # Run Environments
        for i in range(flags.n_envs):
            state = envs_states[i]
            obs = obs_preprocessing(torch.tensor(state, device=flags.device)).unsqueeze(0)
            if mode != 2:
                domains = obs_preprocessing(torch.tensor(get_domains(envs[i]), dtype=torch.float32, device=flags.device)).unsqueeze(0)
            else:
                with torch.no_grad():
                    if len(envs_history[i]['states']) < 1:
                        state_history = torch.zeros((1, envs[0].observation_space.shape[0]))
                    else:
                        state_history = torch.Tensor(envs_history[i]['states'][-flags.len_history:])
                    if len(envs_history[i]['actions']) < 1:
                        onehot_action_history = torch.zeros((1, envs[0].action_space.n))
                    else:
                        action_history = torch.Tensor(envs_history[i]['actions'][-flags.len_history:])
                        onehot_action_history = nn.functional.one_hot(action_history.long(), envs[0].action_space.n).squeeze(-2).to(torch.float32)
                    encoded_states = agent.state_encoder(state_history.unsqueeze(0).to(flags.device))
                    encoded_actions = agent.action_encoder(onehot_action_history.unsqueeze(0).to(flags.device))
                    history = torch.stack((encoded_states, encoded_actions), dim=2).view(encoded_states.shape[0], -1, encoded_states.shape[-1])[:,:-1,:]
                    domains = agent.OSI(history)[:,-1]
            
            obs_domains = torch.cat((obs, domains), dim=-1)
            
            action = agent.act(obs_domains)
            ns, r, done, info = envs[i].step(action)
            if envs_timesteps[i] > flags.horizon:
                done = True

            envs_history[i]['states'].append(state)
            envs_history[i]['actions'].append([action])
            envs_history[i]['rewards'].append([r])
            envs_history[i]['dones'].append([done])
            envs_history[i]['next_states'].append(ns)
            envs_history[i]['domains'].append(get_domains(envs[i]))

            envs_states[i] = ns
            envs_reward[i] += r
            envs_timesteps[i] += 1
            
            if done:
                buffer.append({k:v for k,v in zip(['states', 'actions', 'rewards',  'dones', 'next_states', 'domains'], 
                            [envs_history[i]['states'], envs_history[i]['actions'], envs_history[i]['rewards'], envs_history[i]['dones'],  envs_history[i]['next_states'], envs_history[i]['domains']])}, device=flags.device)
                for key in ['states', 'actions', 'rewards', 'dones', 'next_states', 'domains']:
                    envs_history[i][key] = []

                returns.append(envs_reward[i])
                episode_steps.append(envs_timesteps[i])

                randomize(envs[i])
                envs_states[i] = envs[i].reset()
                envs_reward[i] = 0
                envs_timesteps[i] = 0

                episode_count += 1
  
    # print("Train is Finished!")
    agent.save(os.path.join(logger.result_path, f"mode{mode}_checkpoint.tar"))
                
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
        "CartPole-v1":lambda x=None : gym.make("CartPole-v1")
    }[flags.env]

    dummy_env = env_class()
    agent = UP_OSI(dummy_env.observation_space, 2, dummy_env.action_space, flags, logger=logger)
    train(env_class, agent, 0, flags, logger)
    train(env_class, agent, 1, flags, logger)
    train(env_class, agent, 2, flags, logger)