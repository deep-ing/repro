### NEED TO BE REIMPLEMENTED: CartPole

import torch
import torch.nn as nn 
import numpy as np 
from utils.losses import (
    compute_model_free_loss,
    compute_system_identification_loss,
)
from utils.construct_model import construct_nn_from_config

class UP_OSI(nn.Module):
    def __init__(self, observation_space, action_space, flags, logger):
        super(UP_OSI, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_input_dim = action_space.n
        self.action_ouput_dim = action_space.n

        self.flags = flags
        self.discount_factor = self.flags.discount_factor
        self.epsilon = self.flags.epsilon_init
        self.epsilon_max_timesteps = self.flags.epsilon_max_timesteps
        self.logger = logger

        self.q_net          = construct_nn_from_config(self.flags.q_net,   self.observation_space.shape[0] + 2, self.action_ouput_dim).to(flags.device)
        self.q_target_net   = construct_nn_from_config(self.flags.q_net,   self.observation_space.shape[0] + 2, self.action_ouput_dim).to(flags.device)

        self.state_encoder = construct_nn_from_config(self.flags.state_encoder, self.observation_space.shape[0], self.flags.hidden_dim).to(flags.device)
        self.action_encoder = construct_nn_from_config(self.flags.action_encoder, self.action_input_dim, self.flags.hidden_dim).to(flags.device)
        self.OSI = construct_nn_from_config(self.flags.OSI, self.flags.hidden_dim, 2).to(flags.device)
        
        self.q_target_net.eval()
        
        self.params = list(self.q_net.parameters()) \
               + list(self.OSI.parameters())
                   
        # self.optimizer = torch.optim.Adam(params, lr=self.flags.learning_rate, 
        #                                   weight_decay=self.flags.weight_decay) 
    
        self.optimizer = torch.optim.RMSprop(self.params, lr=self.flags.learning_rate, 
                                                      weight_decay=self.flags.weight_decay) 
        self.lr_schduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.flags.lr_decay)
        self.update_target() 
    
    def learn(self, batch, mode=0):
        if mode == 0:
            states, actions, rewards, dones, next_states = batch

            # Q Values
            q_values = self.q_net.forward(states).gather(-1, (actions.to(torch.int64)))
            next_q_values = self.q_target_net.forward(next_states).detach().max(dim=-1)[0].unsqueeze(-1)
            next_q_values[dones] = 0

            # --- compute the loss
            loss = compute_model_free_loss(q_values, next_q_values, rewards, self.discount_factor)        
        
        else:
            # System identification
            states, actions, domains = batch

            states = self.state_encoder(states)
            actions = self.action_encoder(actions)
            history = torch.stack((states, actions), dim=2).view(states.shape[0], -1, states.shape[-1])[:,:-1,:]
            domains_pred = self.OSI(history)

            # --- compute the loss
            loss = compute_system_identification_loss(domains_pred, domains)
                
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.logger.log_agent({
            "mf_loss" if mode == 0 else "si_loss": loss.item(),
        })
            
    def act(self, obs):
        if self.flags.random_action:  
            return self.action_space.sample()
        if self.epsilon < np.random.random():
            with torch.no_grad():
                action = self.q_net(obs).argmax(1).item()                
        else:
            action = self.action_space.sample()
        return action 
    
    def update_target(self):
        self.q_target_net.load_state_dict(self.q_net.state_dict())

    def anneal_epsilon(self, timestep):
        self.epsilon = (self.epsilon_max_timesteps - timestep)/self.epsilon_max_timesteps
        self.epsilon = min(self.flags.epsilon_init, max(self.epsilon, self.flags.epsilon_final))
        
    def save(self, path):
        model_state_dicts = []
        model_state_dicts.append(self.q_net.state_dict())
        model_state_dicts.append(self.state_encoder.state_dict())
        model_state_dicts.append(self.action_encoder.state_dict())
        model_state_dicts.append(self.OSI.state_dict())
        torch.save(model_state_dicts, path)
        
    def load(self, path):
        model_state_dicts = torch.load(path)
        self.q_net.load_state_dict(model_state_dicts[0])
        self.state_encoder.load_state_dict(model_state_dicts[1])
        self.action_encoder.load_state_dict(model_state_dicts[2])
        self.OSI.load_state_dict(model_state_dicts[3])
