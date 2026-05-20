import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQNAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

agent = DQNAgent().cuda()
print(agent)

# Test it - give it a fake observation
fake_observation = torch.randn(1, 4).cuda()
output = agent(fake_observation)
print(f"\nInput:  {fake_observation}")
print(f"Output: {output}")
print(f"Output shape: {output.shape}")

from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32).cuda(),
            torch.tensor(actions).cuda(),
            torch.tensor(rewards, dtype=torch.float32).cuda(),
            torch.tensor(np.array(next_states), dtype=torch.float32).cuda(),
            torch.tensor(dones, dtype=torch.float32).cuda()
        )

    def __len__(self):
        return len(self.buffer)

buffer = ReplayBuffer()
print(f"Buffer created, capacity: {buffer.buffer.maxlen}")
print(f"Buffer size: {len(buffer)}")