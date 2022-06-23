# reference: https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import numpy as np

from utils.losses import (
    compute_model_free_loss,
    compute_system_identification_loss,
)
from utils.construct_model import construct_nn_from_config

class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, flags, logger):
        super(PPO, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.flags = flags
        self.logger = logger

        self.gamma = flags.gamma
        self.eps_clip = flags.eps_clip
        self.K_epochs = flags.K_epochs

        self.has_continuous_action_space = flags.has_continuous_action_space

        if self.has_continuous_action_space:
            self.action_var = torch.full((action_dim,), flags.action_std_init * flags.action_std_init).to(flags.device)

        self.actor = construct_nn_from_config(self.flags.actor, self.state_dim, self.action_dim).to(flags.device)
        self.critic = construct_nn_from_config(self.flags.critic, self.state_dim, 1).to(flags.device)

        self.actor_old = construct_nn_from_config(self.flags.actor, self.state_dim, self.action_dim).to(flags.device)
        self.critic_old = construct_nn_from_config(self.flags.critic, self.state_dim, 1).to(flags.device)
        self.update_old()

        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': flags.lr_actor},
            {'params': self.critic.parameters(), 'lr': flags.lr_critic},
        ])

        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.flags.lr_decay)

        self.Mseloss = nn.MSELoss()
    
    def evaluate(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.flags.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy
    
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.flags.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")

    def act(self, state):
        if self.has_continuous_action_space:
            action_mean = self.actor_old(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor_old(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.flags.device)
            action, action_logprob = self.act(state)

        return action, action_logprob
        
    def learn(self, batch):
        states = torch.cat(batch.state)
        actions = torch.cat(batch.action)
        rewards = torch.cat(batch.reward)
        is_terminals = batch.done
        logprobs = torch.cat(batch.logprob)

        discounted_rewards = []
        if is_terminals[-1]:
            discounted_reward = 0
        else:
            discounted_reward = self.critic_old(states[-1]).item()
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(self.flags.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        print(states.shape)
        old_states = states.detach()
        old_actions = actions.detach()
        old_logprobs = logprobs.detach()

        for _ in range(self.K_epochs):
            # Evaluate old states and actions
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

            # Match state_values dim with rewards
            state_values = torch.squeeze(state_values)

            # Ratio of pi_theta:pi_theta_old
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Compute surrogate loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Final loss of clipped PPO objective
            loss = -torch.min(surr1, surr2) + 0.5*self.Mseloss(state_values, rewards) - 0.01*dist_entropy
            loss = loss.mean()

            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.logger.log_agent({
                "ppo_loss": loss.item(),
            })
        
        self.update_old()
    
    def update_old(self):
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())
        
    def save(self, path):
        model_state_dicts = []
        model_state_dicts.append(self.actor_old.state_dict())
        model_state_dicts.append(self.critic_old.state_dict())
        torch.save(model_state_dicts, path)
        
    def load(self, path):
        model_state_dicts = torch.load(path)
        self.actor.load_state_dict(model_state_dicts[0])
        self.critic.load_state_dict(model_state_dicts[1])
        self.update_old()
