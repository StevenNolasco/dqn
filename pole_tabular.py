from agent import PoleAgent

import math
import gymnasium as gym
from tqdm import tqdm

# Training hyperparameters
LEARNING_RATE = 0.01        # How fast to learn (higher = faster but less stable)
N_EPISODES = 100_000        # Number of hands to practice
START_EPSILON = 1.0         # Start with 100% random actions
EPSILON_DECAY = START_EPSILON / (N_EPISODES / 2)  # Reduce exploration over time
FINAL_EPSILON = 0.1         # Always keep some exploration

def quantise(val: float, min_val: float, max_val: float, bins: int):
    # clamp val
    val = max(min(val, max_val), min_val)
    val = (val - min_val) / (max_val - min_val)
    return math.floor(val * bins)

def convert_obs(raw_obs):
    cart_pos = raw_obs[0]
    cart_vel = raw_obs[1]
    pole_ang = raw_obs[2]
    pole_ang_vel = raw_obs[3]

    return (
        quantise(cart_pos, -2.4, 2.4, 4),
        quantise(cart_vel, -5, 5, 5),
        quantise(pole_ang, -.2095, .2095, 10),
        quantise(pole_ang_vel, -.1, .1, 10),
    )

def main():
    # TRAIN
    env = gym.make("CartPole-v1")
    agent = PoleAgent(env, LEARNING_RATE, START_EPSILON, EPSILON_DECAY, FINAL_EPSILON)

    for episode in tqdm(range(N_EPISODES)):
        obs, info = env.reset()
        obs = convert_obs(obs)

        episode_ended = False
        while not episode_ended:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = convert_obs(next_obs)
            agent.update(obs, action, reward, terminated, next_obs)
            episode_ended = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()
    env.close()

    # SHOW
    env = gym.make("CartPole-v1", render_mode="human")
    while True:
        obs, info = env.reset()
        obs = convert_obs(obs)

        episode_ended = False
        while not episode_ended:
            action = agent.get_action(obs, False)
            obs, reward, terminated, truncated, info = env.step(action)
            obs = convert_obs(obs)
            episode_ended = terminated or truncated

if __name__ == "__main__":
    main()
