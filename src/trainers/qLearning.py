""""""""""""""""""""
#  Uses Q learning (reinforcement learning component) to update a Q table to associate
#  actions of buying and selling with states of alphas at multiple horizons, then tests
#  on unseen data.
""""""""""""""""""""
import sys

import numpy as np
import pandas as pd
import torch

import config as conf
from utils import Environment, RegressionNetwork

INSTRUMENT = None 


def get_state_in_Q_table_index(state, min_state_value_with_data,
                               max_state_value_with_data):
    """
    Returns the indices of the states in the table when given alpha values.
    
    Args:
            state: states of the alphas to be indexed
            min_state_value_with_data: an np array of minimum state values
            max_state_value_with_data: an np array of maximum state values
    Returns:
            the indices representing the state in the Q table
    """
    state_space_per_q_table_bucket = (max_state_value_with_data - \
                                      min_state_value_with_data) / conf.Q_SIZE_PER_BUCKET
    state_index_for_q_table = np.trunc((np.array(state) - min_state_value_with_data) / \
                                       state_space_per_q_table_bucket)
    state_index_for_q_table[state_index_for_q_table == conf.Q_SIZE_PER_BUCKET] = \
        conf.Q_SIZE_PER_BUCKET - 1
    return tuple(map(int, state_index_for_q_table.tolist()))


def run_test(env, q_table, validation):
    """
    Iterates through the validation dataset or the test dataset.
    
    Args:
            env: the backtesting environment
            q_table: the Q table holding the Q values for all the states
            validation: boolean to use the validation set
    Returns:
            the total reward per episode and the maximum drawdown per episode
    """
    min_state_value_with_data = conf.Q_MIN_STATE_VALUE_TRAINING[INSTRUMENT]
    max_state_value_with_data = conf.Q_MAX_STATE_VALUE_TRAINING[INSTRUMENT]
    
    rewards_per_episode = []
    drawdown_per_episode = []
    # Iterate through dataset
    while (next_state := env.next_ep("validation" if validation
                                     else "testing")[0]) is not None:
        done = False
        previous_state_index = None
        previous_action = None
        previous_reward = 0
        total_episode_reward = 0
        total_drawdown = 0
        # Run daily data
        while not done:
            # Get state index
            current_state_index = get_state_in_Q_table_index(next_state, min_state_value_with_data,
                                                             max_state_value_with_data)
            # If state index is out of range, make a random action or stick with previous action
            if min(current_state_index) < 0 or max(current_state_index) >= q_table.shape[0]:
                action = previous_action if previous_action != None else env.action_space_sample()
            else:
                # Get action according to the model
                action = np.argmax(q_table[current_state_index][previous_action]) \
                    if previous_action != None else \
                    np.argmax(q_table[current_state_index][0] + q_table[current_state_index][1])
            previous_action = action
            # Execute the action
            next_state, _, previous_reward, done = env.step(action)
            total_episode_reward += previous_reward
            total_drawdown = min(total_drawdown, total_episode_reward)
        rewards_per_episode.append(total_episode_reward)
        drawdown_per_episode.append(total_drawdown)
    return rewards_per_episode, drawdown_per_episode


def run(data_path, parameters={"LEARNING_RATE" : conf.RL_LEARNING_RATE,
                               "DISCOUNTED_FUTURE_REWARD_FACTOR":
                               conf.RL_DISCOUNTED_FUTURE_REWARD_FACTOR},
        save_model=True, verbose=True):
    """
    Runs the Q Learning algorithm.
    
    Args:
            data_path: path to the OFI data
            parameters: hyperparameters for the model
            save_model: saves the Q table to disk
            verbose: prints updates to the command line
    Returns:
            the total reward per episode, the maximum drawdown per episode for
            both validation and testing datasets
    """
    global INSTRUMENT
    # Load data
    if verbose:
        print("Loading data")
    INSTRUMENT = data_path.split('/')[-1].split('.')[0]
    q_table = np.zeros((conf.Q_SIZE_PER_BUCKET,) * conf.HORIZON + (2,2,))

    min_state_value_with_data = conf.Q_MIN_STATE_VALUE_TRAINING[INSTRUMENT]
    max_state_value_with_data = conf.Q_MAX_STATE_VALUE_TRAINING[INSTRUMENT]
    env = Environment(data_path)
    
    exploration_rate = conf.RL_EXPLORATION_RATE
    rewards_per_episode = []
    drawdown_per_episode = []
    # Train model
    if verbose:
        print("Training")
    for episode in range(conf.RL_NUMBER_OF_EPISODES):
        total_episode_reward = 0
        total_drawdown = 0
        current_state = env.next_ep("training")[0]
        done = False
        previous_state_index = None
        previous_action = None
        previous_action_2 = None
        previous_reward = 0
        # Run daily data
        while not done:
            # Get state index
            current_state_index = get_state_in_Q_table_index(current_state,
                                                             min_state_value_with_data,
                                                             max_state_value_with_data)
            # If state index is out of range, skip timestep
            if min(current_state_index) < 0 or max(current_state_index) >= \
                    conf.Q_SIZE_PER_BUCKET:
                current_state, _, previous_reward, done = env.step(None)
                continue
            else:
                # Make random action if need to explore
                if np.random.uniform(0, 1) < exploration_rate:
                    action = env.action_space_sample()
                else:
                    # Get action according to the model
                    action = np.argmax(q_table[current_state_index][previous_action]) \
                        if previous_action != None else env.action_space_sample()
            # Update Q table
            if previous_action_2 != None:
                q_table[previous_state_index][previous_action_2][previous_action] = \
                    (1-parameters["LEARNING_RATE"]) * \
                    q_table[previous_state_index][previous_action_2][previous_action] + \
                    parameters["LEARNING_RATE"]*(previous_reward + \
                                                 parameters["DISCOUNTED_FUTURE_REWARD_FACTOR"] \
                                                 [INSTRUMENT] * np.max(q_table[current_state_index] \
                                                                       [previous_action]))
            # Update metrics for next iteration
            previous_state_index = current_state_index
            previous_action_2 = previous_action
            previous_action = action
            # Execute the action
            current_state, _, previous_reward, done = env.step(action)
            total_episode_reward += previous_reward
            total_drawdown = min(total_drawdown, total_episode_reward)
        if verbose:
            print(f"Epoch: {episode + 1}\t\tTraining Total Reward: {total_episode_reward}")
        # Update exploration rate
        exploration_rate = max(conf.RL_MIN_EXPLORATION_RATE, exploration_rate * \
                               conf.RL_EXPLORATION_DECAY)
        rewards_per_episode.append(total_episode_reward)
        drawdown_per_episode.append(total_drawdown)
    # Validation
    if verbose:
        print("Validation")
    rewards_per_episode_v, drawdown_per_episode_v = run_test(env, q_table, validation=True)
    if verbose:
        print(f"Validation Rewards: {rewards_per_episode_v}")
        print(f"Validation Drawdown: {drawdown_per_episode_v}")
    # Testing
    if verbose:
        print("Testing")
    rewards_per_episode_t, drawdown_per_episode_t = run_test(env, q_table, validation=False)
    if verbose:
        print(f"Testing Rewards: {rewards_per_episode_t}")
        print(f"Testing Drawdown: {drawdown_per_episode_t}")
    # Save the model
    if save_model:
        torch.save(q_table, "../models/q_learning/" + INSTRUMENT + ".pt")
    return rewards_per_episode_v, drawdown_per_episode_v, rewards_per_episode_t, drawdown_per_episode_t
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m trainers.qLearning path_to_ofi_data")
        sys.exit(0)
    run(sys.argv[1])
