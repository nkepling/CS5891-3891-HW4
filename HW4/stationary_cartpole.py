import gymnasium as gym
import numpy as np
from copy import deepcopy
from MCTS import MCTS
from utils import type_checker
import random
from tqdm import tqdm

"""Test the MCTS implementation on a stationary CartPole environment.
"""

def make_env():
    env = gym.make("CartPole-v1")
    return env


def run_episode(env,seed=0):


    observation,info = env.reset()

    done = False
    truncated = False   

    reward_list = []

    while not done and not truncated:

        # We will rebuild the MCTS tree after each decision epoch.  
        mcts_agent = MCTS(env,
                    state=observation,
                    d=15,
                    m=50,
                    c=1.44,
                    gamma=0.999)
        
        best_action, action_values = mcts_agent.search()
        observation, reward, done, truncated,info = env.step(best_action)
        observation, reward = type_checker(observation, reward)
        reward_list.append(reward)

    return sum(reward_list),reward_list


def main():

    num_episodes = 25

    env = make_env()

    episode_rewards = []
    
    print("Starting Episodes")
    for i in tqdm(range(num_episodes), desc="Running Episodes", unit="episode"):
        total_reward,rewards = run_episode(env,seed=i)
        episode_rewards.append(total_reward)

    print("Done")

    mean_reward = np.mean(episode_rewards)
    std_err = np.std(episode_rewards)/np.sqrt(num_episodes)
    print(f"Average reward over {num_episodes} episodes: {mean_reward} +/- {std_err}")

    return mean_reward,std_err


if __name__ == "__main__":
    mean_reward,std_err = main()