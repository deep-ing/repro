from agent.CRAR  import CRAR 
from omegaconf import OmegaConf
from envs import SimpleMazeEnv, MazeEnv, CatcherEnv
from utils.buffer import RolloutBuffer
from utils.logger import PlatformLogger
import os 
import gym 
import datetime
import torch 

flags = OmegaConf.load("configs/config.yml")
output_dir = "results/"

env_class = {
                "simple_maze" :SimpleMazeEnv(),
                "catcher" : CatcherEnv(),
                "maze":MazeEnv(),
                "CartPole-v1":gym.make("CartPole-v1")
            }[flags.env]

envs = [env_class for i in range(flags.n_envs)] 
envs_states = [env.reset() for env in envs]
envs_dones = [False for i in range(flags.n_envs)]
episode_reward = [0 for i in range(flags.n_envs)]
envs_timesteps = [0 for i in range(flags.n_envs)]

obs_preprocessing = lambda x : x.unsqueeze(0)
buffer = RolloutBuffer(flags.buffer_len, ['state', 'action', 'reward',  'done', 'next_state'], obs_preprocessing)

date_now = datetime.datetime.now().__str__()
level1 = datetime.datetime.now().strftime(format="%y-%m-%d")
level2 = datetime.datetime.now().strftime(format="%H-%M-%S")
RESULT_path = os.path.join("results", level1, level2)

if not os.path.isdir(RESULT_path):
    os.makedirs(RESULT_path)

OmegaConf.save(config=flags, f=os.path.join(RESULT_path, "config.yaml"))
logger = PlatformLogger(RESULT_path)
agent = CRAR(envs[0].observation_space, envs[0].action_space, flags, logger=logger)

timestep = 0 
episode_count = 0 
returns = []
episode_steps = []

target_update_count = 0
while timestep < flags.timesteps:
    timestep += flags.n_envs 
    if len(buffer) >= flags.batch_size:
        env_batch = buffer.sample(flags.batch_size, flags.device)
        agent.learn(env_batch)
        
        if timestep % flags.log_freq  == 0:
            info_dict = {"timestep": float(timestep),
                         "target_update_count" : float(target_update_count),
                        "episode_count" : float(episode_count),
                        "epsilon" : float(agent.epsilon)}
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
            
    agent.anneal_epsilon(timestep)
    
    # Run Environments
    envs_states = [envs_states[i] if not envs_dones[i] else envs[i].reset() for i in range(len(envs_states))]
    actions = [agent.act(obs_preprocessing(torch.tensor(state, device=flags.device)).unsqueeze(0)) for state in envs_states]
    steps = [envs[i].step(actions[i]) for i in range(len(envs))]
    for i, (ns, r, done, info) in enumerate(steps):
        buffer.append({k:v for k,v in zip(['state', 'action', 'reward',  'done', 'next_state'], 
                                            [envs_states[i], actions[i], r, done,  ns])}, device=flags.device)
        envs_states[i] = ns
        episode_reward[i] += r
        envs_timesteps[i] += 1
        if envs_timesteps[i] > flags.horizon:
            done = True
        envs_dones[i] = done
        if done:
            episode_count +=1 
            returns.append(episode_reward[i])
            episode_steps.append(envs_timesteps[i])
            episode_reward[i] = 0
            envs_timesteps[i] = 0
            