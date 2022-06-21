
import torch 
import torch.nn as nn 


def construct_nn_from_config(layer_list, input_shape, output_shape):
    nets = []
    for i, layer in enumerate(layer_list) :
        if not isinstance(layer, str):
            if layer[0] in ['Linear', 'Conv2d']:
                (layer_name, in_dim, out_dim, kwargs) = layer
                if i == 0 and in_dim == "auto":
                    in_dim = input_shape
                if i == len(layer_list) - 1 and out_dim == "auto":
                    out_dim = output_shape
                net = getattr(nn, layer_name)(in_dim, out_dim, **kwargs)
            elif layer[0] in ["MaxPool2d", 'Flatten']:
                net = getattr(nn, layer[0])(**layer[1])
        else:
            net = getattr(nn, layer)()
        nets.append(net)
    return nn.Sequential(*nets)
    
if __name__ == "__main__":
    from omegaconf import OmegaConf
    config = OmegaConf.load("config.yml")
    q_net = construct_nn_from_config(config.q_net, 20, 20)
    print(q_net)
    