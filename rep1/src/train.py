
from agent.CRAR  import CRAR 
from omegaconf import OmegaConf
from envs.simple_maze import SimpleMazeEnv
from utils.buffer import RolloutBuffer
from utils.logger import PlatformLogger
import os 
import gym 
import datetime
import torch 

flags = OmegaConf.load("config.yml")
output_dir = "results/"

env = SimpleMazeEnv()
agent = CRAR(env.observation_space, env.action_space, flags)
buffer = RolloutBuffer(flags.buffer_len, ['state', 'action', 'reward',  'done', 'next_state'])

date_now = datetime.datetime.now().__str__()
level1 = datetime.datetime.now().strftime(format="%y-%m-%d")
level2 = datetime.datetime.now().strftime(format="%H-%M-%S")
RESULT_path = os.path.join("results", level1, level2)

if not os.path.isdir(RESULT_path):
    os.makedirs(RESULT_path)

OmegaConf.save(config=flags, f=os.path.join(RESULT_path, "config.yaml"))
logger = PlatformLogger(RESULT_path)


envs = [gym.make("CartPole-v1") for i in range(flags.n_envs)] 
envs_states = [env.reset() for env in envs]
envs_dones = [False for i in range(flags.n_envs)]
episode_reward = [0 for i in range(flags.n_envs)]

timestep = 0 
episode_count = 0 
sum_return = 0
while timestep < flags.timesteps:
    timestep +=1 
    if len(buffer) >= flags.batch_size:
        env_batch = buffer.sample(flags.batch_size, flags.device)
        random_batch1 = buffer.sample(flags.batch_size, flags.device)
        random_batch2 = buffer.sample(flags.batch_size, flags.device)
        
        agent.learn(env_batch, random_batch1, random_batch2)
        epsiode_return_mean = 0 if episode_count==0  else sum_return / episode_count
        
        if timestep % flags.platform_log_freq  == 0:
            info_dict = {"episode_return_mean" : f"{epsiode_return_mean:.3f}"
                        ,"epsilon" : f"{agent.epsilon:.3f}"}
            logger.log_iteration(info_dict)
        if timestep % flags.target_update_freq == 0:
            agent.update_target()

        # Run Environments
    envs_states = [envs_states[i] if not envs_dones[i] else envs[i].reset() for i in range(len(envs_states))]
    a = [agent.act(torch.tensor(state, device=flags.device).unsqueeze(0)) for state in envs_states]
    steps = [envs[i].step(a[i]) for i in range(len(envs))]
    for i, (ns, r, done, info) in enumerate(steps):
        buffer.append({k:v for k,v in zip(['state', 'action', 'reward',  'done', 'next_state'], 
                                            [envs_states[i], a[i], r, done,  ns])}, device=flags.device)
        envs_states[i] = ns
        envs_dones[i] = done
        episode_reward[i] += r
        if done:
            episode_count +=1 
            sum_return += episode_reward[i]
            episode_reward[i] = 0