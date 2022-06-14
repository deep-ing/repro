import torch 
import torch.nn as nn 
from model.q_net import QNet

class CRAR():
    def __init__(self, observation_space, action_space, flags):
        self.observation_space = observation_space
        self.action_space = action_space
        self.flags = flags
        self.beta = self.flags.beta 
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
    
    def learn(self, batch, replay_buffer):
        # Model Free
        states, rewards, dones, infos, next_states = batch
        # update the paramters
    
        mf_loss = None 
        reward_loss = None 
        transition_loss = None 
        ld1_loss = None 
        ld1_prime_loss = None 
        ld2_loss = None 
        loss =  None 
        
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
