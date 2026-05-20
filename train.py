import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from agent import DQNAgent, ReplayBuffer

# Setup
env = gym.make("CartPole-v1")
agent = DQNAgent().cuda()
target_net = DQNAgent().cuda()
target_net.load_state_dict(agent.state_dict())

optimiser = torch.optim.Adam(agent.parameters(), lr=0.001)
buffer = ReplayBuffer()

# Hyperparameters
EPISODES = 500
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01
rewards_history = []

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).cuda()
    with torch.no_grad():
        q_values = agent(state_tensor)
    return q_values.argmax().item()

def train_step():
    if len(buffer) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
    current_q = agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q = target_net(next_states).max(1)[0]
    target_q = rewards + GAMMA * next_q * (1 - dones)
    loss = nn.MSELoss()(current_q, target_q.detach())
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

# Training loop
for episode in range(EPISODES):
    state, _ = env.reset()
    state = np.array(state)
    total_reward = 0

    while True:
        action = select_action(state, EPSILON)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(state, action, reward, next_state, done)
        train_step()
        state = np.array(next_state)
        total_reward += reward
        if done:
            break

    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)
    rewards_history.append(total_reward)

    if episode % 10 == 0:
        target_net.load_state_dict(agent.state_dict())
        print(f"Episode {episode} | Reward: {total_reward:.0f} | Epsilon: {EPSILON:.3f}")

# Plot after training finishes
plt.figure(figsize=(10, 5))
plt.plot(rewards_history, alpha=0.4, color='blue', label='Raw rewards')
window = 20
smoothed = [np.mean(rewards_history[max(0, i-window):i+1])
            for i in range(len(rewards_history))]
plt.plot(smoothed, color='red', linewidth=2, label='20 episode average')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DQN CartPole Training')
plt.legend()
plt.savefig('training_curve.png')
print("Graph saved to training_curve.png")

torch.save(agent.state_dict(), "cartpole_agent.pth")
print("Agent saved")
