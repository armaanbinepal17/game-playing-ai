import torch
import gymnasium as gym
import numpy as np

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Gymnasium: {gym.__version__}")
print(f"NumPy: {np.__version__}")

# Create the CartPole environment
env = gym.make("CartPole-v1")
observation, info = env.reset()

print(f"\nCartPole environment created")
print(f"Observation shape: {observation.shape}")
print(f"Observation: {observation}")
print(f"Number of actions: {env.action_space.n}")