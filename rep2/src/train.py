import os
import datetime
from collections import deque

import gym
import numpy as np
import torch

from omegaconf import OmegaConf

from agent import PPO
from agent.envs import make_vec_envs
from agent.utils import update_linear_schedule, get_vec_normalize
from agent.storage import RolloutStorage
from utils.logger import PlatformLogger

def eval(env_name, agent, seed, domain_randomization_dict, flags, logger):
    env = make_vec_envs(env_name, seed + flags.seed + flags.num_envs, 1, None, None, flags, True)

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = agent.obs_rms

    domain_names = list(domain_randomization_dict.keys())
    episode_rewards = [0] * flags.test_num_episodes
    episode_len = [0] * flags.test_num_episodes
    for i in range(flags.test_num_episodes):
        done = False
        randomize(env, domain_randomization_dict)
        domain = get_domain(env, domain_names)

        obs = env.reset()
        eval_recurrent_hidden_states = torch.zeros(1, agent.actor_critic.recurrent_hidden_state_size, device=flags.device)
        eval_masks = torch.zeros(1, 1, device=flags.device)
        for _ in range(flags.test_max_episode_len):
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

            episode_rewards[i] += reward.item()
            episode_len[i] += 1
            
            if done:
                break

    env.close()
            
    info_dict = {
        "episode_return_mean" : np.mean(episode_rewards),
        "episode_return_median" : np.median(episode_rewards),
        "episode_return_var" : np.var(episode_rewards),
        "episode_len_mean": np.mean(episode_len),
    }
    logger.log_eval(info_dict)
    
# Put this code to envs.
# def get_simulator_parameters(self, keys): ####
#     return [getattr(self, key)/getattr(self, key+'_original') if hasattr(self, key+'_original') else 1 for key in keys]

# def set_simulator_parameters(self, params): ####
#     # params: dict with (param_name, scaling_factor)
#     for key in params:
#         if not hasattr(self, key+'_original'):
#             setattr(self, key+'_original', getattr(self, key))
#         setattr(self, key, params[key]*getattr(self, key+'_original'))

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

def train(env_name, agent, domain_randomization_dict, flags, RESULT_path, logger):
    envs = make_vec_envs(env_name, flags.seed, flags.num_envs, flags.discount_factor, RESULT_path, flags, False)

    domain_names = list(domain_randomization_dict.keys())

    rollouts = RolloutStorage(flags.num_steps, flags.num_envs, envs.observation_space.shape, envs.action_space, agent.actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(flags.device)

    # keep data for logging
    episode_rewards = deque(maxlen=10)
    episode_len = deque(maxlen=10)
    num_samples = 0
    num_episodes = 0

    num_updates = flags.max_training_steps // flags.num_steps // flags.num_envs
    for j in range(num_updates):
        if flags.use_linear_lr_decay:
            update_linear_schedule(agent.optimizer, j, num_updates, flags.lr)

        # Collect data
        for step in range(flags.num_steps):
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = agent.actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_len.append(info['episode']['l'])
                    num_episodes += 1
            
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        num_samples += flags.num_envs*flags.num_steps

        # Compute returns
        with torch.no_grad():
            next_value = agent.actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, flags.use_gae, flags.discount_factor,
                                flags.gae_lambda, flags.use_proper_time_limits)

        # Update model
        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # Save model
        if j % flags.save_freq == 0:
            agent.save(getattr(get_vec_normalize(envs), 'obs_rms', None),
                os.path.join(logger.result_path, f"checkpoint_{j}.tar"))
            print("[INFO] Checkpoint is saved")

        # Logging
        if j % flags.log_freq == 0:
            info_dict = {
                "num_samples": float(num_samples),
                "num_episodes": float(num_episodes),
                "value_loss": value_loss,
                "action_loss": action_loss,
                "dist_entropy": dist_entropy,
            }
            if len(episode_rewards) > 0:
                info_dict.update({"episode_return_mean": np.mean(episode_rewards)})
            if len(episode_len) > 0:
                info_dict.update({"episode_len_mean": np.mean(episode_len)})
            logger.log_iteration(info_dict)

        # Evaluation
        if j % flags.eval_freq == 0:
            agent.obs_rms = get_vec_normalize(envs).obs_rms
            eval(env_name, agent, j // flags.eval_freq, domain_randomization_dict, flags, logger)
            print("[INFO] Evaluation is done")
   
    print("Train is Finished!")
    agent.save(getattr(get_vec_normalize(envs), 'obs_rms', None),
        os.path.join(logger.result_path, f"checkpoint.tar"))
                
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

    torch.manual_seed(flags.seed)
    torch.cuda.manual_seed_all(flags.seed)
    np.random.seed(flags.seed)

    torch.set_num_threads(1)

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
    
    train(flags.env_name, agent, domain_randomization_dict, flags, RESULT_path, logger)