
import torch 
import torch.nn as nn 

def construct_nn_from_config(layer_list, input_shape, output_shape):
    nets = []
    for layer_name, kwargs in layer_list :
        net = None 
        
        nets.append(net) 
    return nn.Squential(*nets)
    
    
    