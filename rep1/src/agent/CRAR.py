import torch 
import torch.nn as nn 
import numpy as np 
from utils.losses import (
    compute_LD1_loss,
    compute_LD1_prime_loss,
    compute_LD2_loss,
    compute_model_free_loss,
    compute_reward_loss,
    compute_transition_loss
)
from utils.construct_model import construct_nn_from_config

class CRAR(nn.Module):
    def __init__(self, observation_space, action_space, flags, logger):
        super(CRAR, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_input_dim = 1
        self.action_ouput_dim = action_space.n
        self.flags = flags
        self.beta = self.flags.beta 
        self.ld_lambda = self.flags.ld_lambda
        self.model_lambda = self.flags.model_lambda
        self.discount_factor = self.flags.discount_factor
        self.abstract_dim = self.flags.abstract_dim 
        self.epsilon = self.flags.epsilon_init
        self.epsilon_max_timesteps = self.flags.epsilon_max_timesteps
        self.logger = logger
        
        self.encoder        = construct_nn_from_config(self.flags.encoder, 1, self.abstract_dim).to(flags.device)
        self.encoder_target = construct_nn_from_config(self.flags.encoder, 1, self.abstract_dim).to(flags.device)
        self.q_net          = construct_nn_from_config(self.flags.q_net,   self.abstract_dim, self.action_ouput_dim).to(flags.device)
        self.q_target_net   = construct_nn_from_config(self.flags.q_net,   self.abstract_dim, self.action_ouput_dim).to(flags.device)
        self.transition_net = construct_nn_from_config(self.flags.transition_net, self.abstract_dim+self.action_input_dim, self.abstract_dim).to(flags.device)
        self.reward_net     = construct_nn_from_config(self.flags.reward_net,     self.abstract_dim+self.action_input_dim, 1).to(flags.device)
        self.discount_net   = construct_nn_from_config(self.flags.discount_net,   self.abstract_dim+self.action_input_dim, 1).to(flags.device)
        self.q_target_net.eval()
        
        self.params = list(self.q_net.parameters()) \
               + list(self.encoder.parameters()) \
               + list(self.transition_net.parameters()) \
               + list(self.reward_net.parameters()) \
               + list(self.discount_net.parameters())
                   
        # self.optimizer = torch.optim.Adam(params, lr=self.flags.learning_rate, 
        #                                   weight_decay=self.flags.weight_decay) 
    
        self.optimizer = torch.optim.RMSprop(self.params, lr=self.flags.learning_rate, 
                                                      weight_decay=self.flags.weight_decay) 
        self.lr_schduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.flags.lr_decay)
        self.update_target() 
    
    def learn(self, batch):
        states, actions, rewards, dones, next_states = batch
        # random_states2 = states
        encoded_states  = self.encoder(states)
        encoded_next_states = self.encoder_target(next_states)
        encoded_random_states1 =  encoded_states                 # self.encoder(random_states1)
        encoded_random_states2 =  encoded_states.roll(1, dims=0) # self.encoder(random_states2)
        # Q Values 
        q_values = self.q_net.forward(encoded_states).gather(-1, (actions.to(torch.int64)))
        next_q_values = self.q_target_net.forward(encoded_next_states).detach().max(dim=-1)[0].unsqueeze(-1)
        next_q_values[dones] = 0         
        # Transition, Reward, Discount 
        assert actions.ndim==2, actions.ndim
        transition_pred = self.transition_net(torch.cat((encoded_states, actions), 1))
        reward_pred     = self.reward_net(torch.cat((encoded_states, actions), 1))
        # discount_pred   = self.discount_net(torch.cat((encoded_states, actions), 1))

        # --- compute the loss
        mf_loss         = compute_model_free_loss(q_values, next_q_values, rewards, self.discount_factor)        
        reward_loss     = compute_reward_loss(reward_pred, rewards) 
        transition_loss = compute_transition_loss(transition_pred + encoded_states, encoded_next_states)
        # discount_loss   = compute_discount_loss()
        ld1_loss        = compute_LD1_loss(encoded_random_states1, encoded_random_states2) 
        ld1_prime_loss  = compute_LD1_prime_loss(encoded_states, encoded_next_states) 
        ld2_loss        = compute_LD2_loss(encoded_states)
        ld_loss = ld1_loss + self.beta * ld1_prime_loss + ld2_loss
        loss =  mf_loss + self.model_lambda * (reward_loss + transition_loss) + self.ld_lambda * ld_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.logger.log_agent({
            "mf_loss" : mf_loss.item(),
            "reward_loss" : reward_loss.item(),
            "transition_loss" : transition_loss.item(),
            "ld1_loss" : ld1_loss.item(),
            "ld1_prime_loss" : ld1_prime_loss.item(),
            "ld2_loss" : ld2_loss.item(),
            "ld_loss" : ld_loss.item(),
            "total_loss" : loss.item(),
        })
    
    def act(self, obs):
        if self.flags.random_action:  
            return self.action_space.sample()
        if self.epsilon < np.random.random():
            with torch.no_grad():    
                obs = self.encoder(obs)
                action = self.q_net(obs).argmax(1).item()                
        else:
            action = self.action_space.sample()
        return action 
    
    def update_target(self):
        self.q_target_net.load_state_dict(self.q_net.state_dict())
        self.encoder_target.load_state_dict(self.encoder.state_dict())

    def anneal_epsilon(self, timestep):
        self.epsilon = (self.epsilon_max_timesteps - timestep)/self.epsilon_max_timesteps
        self.epsilon = min(self.flags.epsilon_init, max(self.epsilon, self.flags.epsilon_final))
        
    def save(self, path):
        model_state_dicts = []
        model_state_dicts.append(self.encoder.state_dict())
        model_state_dicts.append(self.q_net.state_dict())
        model_state_dicts.append(self.reward_net.state_dict())
        model_state_dicts.append(self.discount_net.state_dict())
        model_state_dicts.append(self.transition_net.state_dict())
        torch.save(model_state_dicts, path)
        
    def load(self, path):
        model_state_dicts = torch.load(path)
        self.encoder.load_state_dict(model_state_dicts[0])
        self.q_net.load_state_dict(model_state_dicts[1])
        self.reward_net.load_state_dict(model_state_dicts[2])
        self.discount_net.load_state_dict(model_state_dicts[3])
        self.transition_net.load_state_dict(model_state_dicts[4])
