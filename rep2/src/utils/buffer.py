from collections import deque 
import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, maxlen, value_name_list, obs_preprocessing=None):
        self.maxlen = maxlen
        self.values_to_stroe = value_name_list
        self.deques = {}
        for value in self.values_to_stroe:
            self.deques[value] = deque(maxlen=self.maxlen)
        self.obs_preprocessing = obs_preprocessing
        
    def append(self, values, device):
        assert set(values.keys()) == set(self.values_to_stroe)
        for key, value in values.items():
            if (isinstance(value, np.ndarray)) :
                value = self.obs_preprocessing(torch.Tensor(value).to(device))
            elif (isinstance(value, list)) :
                value = self.obs_preprocessing(torch.Tensor(value).to(device))
            elif isinstance(value, (float)) :
                value = torch.Tensor([value]).to(device)
            elif isinstance(value, (bool)) :
                value = torch.BoolTensor([value]).to(device)
            elif isinstance(value, (int)) :
                value = torch.Tensor([value]).to(device)
            self.deques[key].append(value)
    
    def update_discount_return(self, count_episode, discount_factor):
        # compute the discount return from the last done until the next done 
        # when the number of finished episode is 1, compute to to the start. 
        start, end = None, None 
        if count_episode == 1:
            start, end = 0, len(self['done'])-1
        else:
            last = len(self['done'])
            num_find = 0 
            for i in range(last-1, 0, -1):
                if self['done'][i] == True:
                        if num_find ==0:
                            end = i 
                            num_find+=1
                        elif num_find ==1:
                            start = i+1 
                            break
        discount_cumulated = 0
        for i in range(end, start-1, -1):
            discount_cumulated = self['reward'][i] + discount_cumulated * discount_factor
            self['reward'][i] = discount_cumulated
        return 

    def clear_all(self):
        for v in self.deques.values():
            v.clear()
            
    def __getitem__(self, i):
        return self.deques[i]

    def __len__(self):
        return len(self.deques[self.values_to_stroe[0]])

    def sample(self, batch_size, len_history, device):
        indices = np.random.choice(range(len(self)), size=batch_size, replace=True)
        indices2 = np.random.randint(0, [len(self[self.values_to_stroe[0]][i]) - len_history for i in indices])

        batch_dict = []
        for v in self.values_to_stroe:
            try:
                batch_dict.append(torch.stack([self[v][i][j:j+len_history] for i, j in zip(indices, indices2)]).to(device))
            except:
                print(v, self[v][0])    
        return batch_dict


if __name__ == "__main__":
    config = {
        "maxlen" :1000,
        "value_name_list":['action', 'state', 'reward', 'done']
    }
    buffer =RolloutBuffer(**config)
    for i in range(100):
        result = {
            "action": i,
            "state":[1,2+i,3-i],
            "reward":3*i,
            "done":True if i==0 else False
        }
        buffer.append(result, device="cpu")
        # print(buffer.deques)

    batch_dict = buffer.sample(32, device="cpu")
    print(batch_dict['action'])