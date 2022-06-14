
import torch 
import torch.nn as nn 
from model.q_net import QNet
from utils.losses import (
    compute_LD1_loss,
    compute_LD1_prime_loss,
    compute_LD2_loss,
    compute_model_free_loss,
    compute_reward_loss,
    compute_transition_loss
)
class CRAR():
    def __init__(self, observation_space, action_space, flags):
        self.observation_space = observation_space
        self.action_space = action_space
        self.flags = flags
        self.beta = self.flags.beta 
        self.discount_factor = self.flags.discount_factor
        self.abstract_dim = self.flags.abstract_dim 
        
        self.q_net =  nn.Linear(10,20) 
        self.q_target_net = nn.Linear(10,20) 
        self.encoder =  nn.Linear(10,20) 
        self.transition_net = nn.Linear(10,20) 
        self.reward_net = nn.Linear(10,20)  
        self.discount_net = nn.Linear(10,20)
        
        params = list(self.q_net.parameters()) \
               + list(self.encoder.parameters()) \
               + list(self.transition_net.parameters()) \
               + list(self.reward_net.parameters()) \
               + list(self.discount_net.parameters())
                   
        self.optimizer = torch.optim.Adam(params, lr=self.flags.learning_rate, 
                                          weight_decay=self.flags.weight_decay) 


        self.update_target() 
    
    def learn(self, batch, random_batch_1, random_batch_2):
        states, actions, rewards, dones, next_states = batch
        random_states1, _, _, _, _ = random_batch_1
        random_states2, _, _, _, _ = random_batch_2
        
        encoded_states = self.encoder(states)
        encoded_next_states = self.encoder(next_states)
        encoded_random_states1 = self.encoder(random_states1)
        encoded_random_states2 = self.encoder(random_states2)
        
        # Q Values 
        q_values = self.q_net.forward(encoded_states).gather(-1, (actions.to(torch.int64)))
        next_q_values = self.q_target_net.forward(next_states).detach().max(dim=-1)[0].unsqueeze(-1)
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
        
        self.optimzer.zero_grad()
        loss.backward()
        self.optimzer.step()
    
    def act(self, obs):
        if self.flags.random_action:  
            return self.action_space.sample()
        with torch.no_grad():
            action = self.q_net(obs).argmax(1)
        return action 
    
    def update_target(self):
        self.q_target_net.load_state_dict(self.q_net.state_dict())