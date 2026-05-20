import torch
import numpy as np
import gymnasium as gym
from agent import DQNAgent

env = gym.make("CartPole-v1", render_mode="human")
agent = DQNAgent().cuda()
agent.load_state_dict(torch.load("cartpole_agent.pth"))
agent.eval()

for episode in range(5):
    state, _ = env.reset()
    state = np.array(state)
    total_reward = 0

    while True:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda()
        with torch.no_grad():
            action = agent(state_tensor).argmax().item()
        state, reward, terminated, truncated, _ = env.step(action)
        state = np.array(state)
        total_reward += reward
        if terminated or truncated:
            print(f"Episode reward: {total_reward}")
            break

env.close()