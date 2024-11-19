import gymnasium as gym
import ns_gym
import numpy as np
from copy import deepcopy
from MCTS_KEY import MCTS
from utils import type_checker
import random
from tqdm import tqdm

"""Test the MCTS implementation on a stationary CartPole environment.
"""

def make_ns_env(change_notification=False,delta_change_notification=False):
    env = gym.make("CartPole-v1",max_episode_steps=500)
    scheduler = ns_gym.schedulers.ContinuousScheduler()
    update_fn = ns_gym.update_functions.IncrementUpdate(scheduler=scheduler,k=0.1)
    param_name = "length" # pole length
    params = {param_name:update_fn}

    env = ns_gym.wrappers.NSClassicControlWrapper(env,
                                                  params,
                                                  change_notification=change_notification,
                                                  delta_change_notification=delta_change_notification)

    return env


def run_episode(env,seed=0):

    observation,info = env.reset()
    planning_env = env.get_planning_env()

    done = False
    truncated = False   

    reward_list = []

    while not done and not truncated:

        # We will rebuild the MCTS tree after each decision epoch.  
        mcts_agent = MCTS(planning_env,
                    state=observation,
                    d=15,
                    m=50,
                    c=1.44,
                    gamma=0.999)
        
        best_action, action_values = mcts_agent.search()
        observation, reward, done, truncated,info = env.step(best_action)
        planning_env = env.get_planning_env()
        observation, reward = type_checker(observation, reward)
        reward_list.append(reward)

    return sum(reward_list),reward_list


def main():

    num_episodes = 25

    ###### Change the following to True to enable environmental parameter change notification ######
    change_notification = False 
    delta_change_notification = False
    ######################################################################################################

    env = make_ns_env(change_notification,delta_change_notification)

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