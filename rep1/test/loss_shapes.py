

def test_losses():
    import os
    import sys
    PROJECT_PATH = os.getcwd()
    SOURCE_PATH = os.path.join(
        PROJECT_PATH,"src"
    )
    sys.path.append(SOURCE_PATH)
    
    from utils.losses import (
        MSELOSS,
        DISAMBIGUATION,
        compute_model_free_loss,
        compute_transition_loss,
        compute_reward_loss,
        compute_LD1_loss,
        compute_LD1_loss,
        compute_LD1_prime_loss,
        compute_LD2_loss
    )
    import torch 
    num_batches = 32
    actiom_dim = 10
    encoded_dim = 64
    
    state_action_values = torch.rand(32, actiom_dim)
    next_state_action_values = torch.rand(32, actiom_dim)
    transitions = torch.rand(32, encoded_dim)
    encodede_states = torch.rand(32, encoded_dim)
    encodede_next_states = torch.rand(32, encoded_dim)
    rewards = torch.rand(32, 1)
    gamma = 0.99 
    reward_prediction = torch.rand(32, 1)
    encoded_random_states1 = torch.rand(32, encoded_dim)
    encoded_random_states2 = torch.rand(32, encoded_dim)
    
    
    mf_loss = compute_model_free_loss(state_action_values, 
                                      next_state_action_values,
                                      rewards,
                                      gamma)
    transition_loss = compute_transition_loss(transitions, encodede_states)
    reward_loss =compute_reward_loss(reward_prediction, rewards)
    LD1_loss = compute_LD1_loss(encoded_random_states1, encoded_random_states2)
    LD1_prime_loss = compute_LD1_loss(encodede_states, encodede_next_states)
    LD2_loss = compute_LD2_loss(encodede_states)
    print("Model Free Loss : ", mf_loss)
    print("Transition Loss : ", transition_loss)
    print("Reward     Loss : ", reward_loss)
    print("LD1        Loss : ", LD1_loss)
    print("LD1 Prime  Loss : ", LD1_prime_loss)
    print("LD2        Loss : ", LD2_loss)


if __name__ == "__main__":
    test_losses()