

import torch 
import torch.nn as nn 

MSELOSS = nn.MSELoss()

def compute_model_free_loss(state_action_values, next_state_values, rewards, gamma):
    assert state_action_values.size()[1:] ==  next_state_values.size()[1:]
    assert state_action_values.size()[1:] ==  rewards.size()[1:]
    expected_state_action_values = rewards + gamma * next_state_values 
    return nn.SmoothL1Loss()(state_action_values, expected_state_action_values)

def compute_system_identification_loss(pred, gt):
    assert pred.size() == gt.size()
    return nn.MSELoss()(pred, gt)