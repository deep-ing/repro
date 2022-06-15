import torch 
import torch.nn as nn 
from model.encoder import Encoder
import numpy as np 
from utils.losses import (
    compute_LD1_loss,
    compute_LD1_prime_loss,
    compute_LD2_loss,
    compute_model_free_loss,
    compute_reward_loss,
    compute_transition_loss
)
class CRAR():
    def __init__(self, observation_space, action_space, flags, logger):
        self.observation_space = observation_space
        self.action_space = action_space
        self.flags = flags
        self.beta = self.flags.beta 
        self.discount_factor = self.flags.discount_factor
        self.abstract_dim = self.flags.abstract_dim 
        self.epsilon = self.flags.epsilon_init
        self.logger = logger
        
        self.encoder =  Encoder(self.abstract_dim).to(flags.device)
        self.q_net =  nn.Linear(self.abstract_dim, action_space.n).to(flags.device)
        self.q_target_net = nn.Linear(self.abstract_dim, action_space.n).to(flags.device) 
        self.transition_net = nn.Linear(self.abstract_dim,self.abstract_dim).to(flags.device) 
        self.reward_net = nn.Linear(self.abstract_dim, 1).to(flags.device)  
        self.discount_net = nn.Linear(self.abstract_dim, 1).to(flags.device)
        
        params = list(self.q_net.parameters()) \
               + list(self.encoder.parameters()) \
               + list(self.transition_net.parameters()) \
               + list(self.reward_net.parameters()) \
               + list(self.discount_net.parameters())
                   
        self.optimizer = torch.optim.Adam(params, lr=self.flags.learning_rate, 
                                          weight_decay=self.flags.weight_decay) 


        self.update_target() 
    
    def learn(self, batch):
        states, actions, rewards, dones, next_states = batch
        random_states1 = states
        random_states2 = states.roll()
        encoded_states = self.encoder(states)
        encoded_next_states = self.encoder(next_states)
        encoded_random_states1 = self.encoder(random_states1)
        encoded_random_states2 = self.encoder(random_states2)
        
        # Q Values 
        q_values = self.q_net.forward(encoded_states).gather(-1, (actions.to(torch.int64)))
        next_q_values = self.q_target_net.forward(encoded_next_states).detach().max(dim=-1)[0].unsqueeze(-1)
        next_q_values[dones] = 0         
        # Transition, Reward, Discount 
        transition_pred = self.transition_net(encoded_states)
        reward_pred = self.reward_net(encoded_states)
        discount_pred = self.discount_net(encoded_states)

        # --- compute the loss
        mf_loss         = compute_model_free_loss(q_values, next_q_values, rewards, self.discount_factor)        
        reward_loss     = compute_reward_loss(reward_pred, rewards) 
        transition_loss = compute_transition_loss(transition_pred, encoded_states)
        ld1_loss        = compute_LD1_loss(encoded_random_states1, encoded_random_states2) 
        ld1_prime_loss  = compute_LD1_prime_loss(encoded_states, encoded_next_states) 
        ld2_loss        = compute_LD2_loss(encoded_states)
        ld_loss = ld1_loss + self.beta * ld1_prime_loss + ld2_loss
        loss =  mf_loss + reward_loss + transition_loss + ld_loss
        
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
        with torch.no_grad():
            action = self.q_net(obs).argmax(1)
        return action 
    
    def update_target(self):
        self.q_target_net.load_state_dict(self.q_net.state_dict())

    def anneal_epsilon(self):
        self.epsilon *= 0.99
        self.epsilon = max(self.epsilon, self.flags.epsilon_final)