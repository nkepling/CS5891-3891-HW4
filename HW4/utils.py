from ns_gym import base
import numpy as np

def type_checker(observation, reward):
    """Grabs only the the observation and reward from base.Observation and base.Reward type without NS-Gym specific attributes.

    Also converts the observation and reward to tuples if necessary.

    Args:
        observation (ns_gym.base.Observation): Observation to convert.
        reward (ns_gym.base.Reward): Reward to convert.

    Returns:
        (int,np.ndarray): Converted observation.
        (float): Converted reward.
    """
    # Check if observation is not None
    if observation is not None:
        if isinstance(observation, base.Observation):
            observation = observation.state
        if isinstance(observation, np.ndarray):
            observation = tuple(observation)


    # Check if reward is not None
    if reward is not None:
        if isinstance(reward, base.Reward):
            reward = reward.reward

    return observation, reward



