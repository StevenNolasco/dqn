from dqn import PoleAgent

import gymnasium as gym
from tqdm import tqdm
import torch
from itertools import count
import matplotlib.pyplot as plt

# Training hyperparameters
LEARNING_RATE = 3e-4
N_EPISODES = 1_000
START_EPSILON = 0.9
EPSILON_DECAY = START_EPSILON / (N_EPISODES / 2)
FINAL_EPSILON = 0.01
DISCOUNT_FACTOR = 0.99

def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    return torch.device("cuda")

def normalise(val: float, min_val: float, max_val: float):
    # clamp val
    val = max(min(val, max_val), min_val)
    return (val - min_val) / (max_val - min_val)

def convert_obs(raw_obs):
    cart_pos = raw_obs[0]
    cart_vel = raw_obs[1]
    pole_ang = raw_obs[2]
    pole_ang_vel = raw_obs[3]

    return (
        normalise(cart_pos, -2.4, 2.4),
        normalise(cart_vel, -5, 5),
        normalise(pole_ang, -.2095, .2095),
        normalise(pole_ang_vel, -.1, .1),
    )

def plot_durations(episode_durations: list, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def train():
    DEVICE = get_device()

    # TRAIN
    env = gym.make("CartPole-v1")
    obs, _ = env.reset()
    agent = PoleAgent(env, LEARNING_RATE, START_EPSILON, EPSILON_DECAY, FINAL_EPSILON, DISCOUNT_FACTOR, len(obs), DEVICE)

    episode_durations = []

    for episode in tqdm(range(N_EPISODES)):
        obs, _ = env.reset()
        state = torch.tensor(convert_obs(obs), dtype=torch.float32, device=DEVICE).unsqueeze(0)

        for t in count():
            action = agent.get_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=DEVICE)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # Store the transition in memory
            agent.append_memory(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.update()

            if done:
                episode_durations.append(t + 1)
                if episode % 10 == 0:
                    plot_durations(episode_durations)
                break
    env.close()

    plt.savefig("fig.png")
    agent.save()

def show():
    DEVICE = get_device()

    # TRAIN
    env = gym.make("CartPole-v1", render_mode="human")
    obs, _ = env.reset()
    agent = PoleAgent(env, LEARNING_RATE, START_EPSILON, EPSILON_DECAY, FINAL_EPSILON, DISCOUNT_FACTOR, len(obs), DEVICE)
    agent.load()

    while True:
        obs, _ = env.reset()
        state = torch.tensor(convert_obs(obs), dtype=torch.float32, device=DEVICE).unsqueeze(0)

        for t in count():
            action = agent.get_action(state, True)
            observation, _, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            # Move to the next state
            state = next_state

            if done:
                break

if __name__ == "__main__":
    # set up matplotlib
    plt.ion()

    #train()
    show()
