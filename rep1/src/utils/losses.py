
# Based on 
# https://github.com/aayn/crar-pytorch/blob/master/crar/losses.py


import torch 
import torch.nn as nn 

MSELOSS = nn.MSELoss()
DISAMBIGUATION = lambda t1, t2, cd=5.0 : \
    torch.exp(-cd * torch.clamp(torch.norm(t1 - t2, dim=1, keepdim=True), 1e-6, 3.2)).sum()

def compute_model_free_loss(state_action_values, next_state_values, rewards, gamma):
    assert state_action_values.size()[1:] ==  next_state_values.size()[1:]
    assert state_action_values.size()[1:] ==  rewards.size()[1:]
    expected_state_action_values = rewards + next_state_values * gamma
    return MSELOSS(state_action_values, expected_state_action_values)

def compute_transition_loss(transition, encoded_next_states):
    assert transition.ndim == 2 # vector representation
    assert transition.size()[1:] == encoded_next_states.size()[1:]
    return MSELOSS(transition, encoded_next_states)

def compute_reward_loss(reward_predictions, rewards):
    assert reward_predictions.ndim == 2 # vector representation
    assert reward_predictions.size()[1:] == rewards.size()[1:]
    return MSELOSS(reward_predictions, rewards)


def compute_LD1_loss(encoded_random_states1, encoded_random_states2):
    "disambiguation loss between random states."
    assert encoded_random_states1.ndim == 2 # vector representation
    assert encoded_random_states1.size()[1:] == encoded_random_states2.size()[1:]
    return DISAMBIGUATION(encoded_random_states1, encoded_random_states2)

def compute_LD1_prime_loss(encoded_states, encoded_next_states):
    "disambiguation loss between consecutive states."
    assert encoded_states.ndim == 2 # vector representation
    assert encoded_states.size()[1:] == encoded_next_states.size()[1:]
    return DISAMBIGUATION(encoded_states, encoded_next_states)

def compute_LD2_loss(encoded_states):
    return torch.clamp(torch.max(torch.pow(encoded_states, 2))- 1.0, min=0.0, max=100.0)