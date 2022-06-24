from collections import namedtuple, deque
import random

class ReplayMemory:
    def __init__(self, capacity, keys):
        self.memory = deque([], maxlen=capacity)
        self.Transition = namedtuple('Transition', keys)

    def push(self, *args):
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        if batch_size > 0:
            transitions = random.sample(self.memory, batch_size)
        else:
            transitions = self.memory
        batch = self.Transition(*zip(*transitions))
        return batch
        
    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()

if __name__ == "__main__":
    keys = ('state', 'action', 'reward', 'done', 'next_state')
    memory = ReplayMemory(capacity=1000, keys=keys)
    batch_size = 16

    state = [0, 0, 0]
    for i in range(100):
        next_state, action, reward, done = [0, 0, 0], 0, i, False
        memory.push(state, action, reward, done, next_state)
        state = next_state

    print("Sampling before clear:")
    batch = memory.sample(batch_size)
    print(batch)

    memory.clear()
    
    print("\nSampling after clear:")
    try:    
        batch = memory.sample(batch_size)
        print(batch)
    except Exception as e:
        print(e)