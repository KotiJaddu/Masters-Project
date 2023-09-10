""""""""""""""""""""
#  Trains a double deep Q network (reinforcement learning component) to
#  learn when to buy or sell when given alphas at multiple horizons as input,
#  then tests on unseen data.
""""""""""""""""""""
from collections import deque
import random
import sys

import numpy as np
import torch
import torch.optim as optim

import config as conf
from utils import DeepQNetwork, Environment


def update_target_network(policy_net, target_net, target_update_weight):
    """
    Updates the target network using the policy network.
    
    Args:
            policy_net: the policy network
            target_network: the target network
            target_update_weight: the weight of update to the target network (tau)
    """
    policy_weights = policy_net.parameters()
    target_weights = target_net.parameters()
    for policy_weight, target_weight in zip(policy_weights, target_weights):
        updated_weights = target_weight.data * \
            (1 - target_update_weight) + policy_weight.data * target_update_weight
        target_weight.data.copy_(updated_weights)

        
def test_run(env, model, validation):
    """
    Iterates through the validation dataset or the test dataset.
    
    Args:
            env: the backtesting environment
            model: the policy network holding the Q values for the different alpha values
            validation: boolean to use the validation set
    Returns:
            the total reward per episode and the maximum drawdown per episode
    """
    rewards_per_episode = []
    drawdown_per_episode = []
    # Iterate through dataset
    while (current_state := env.next_ep("validation" if validation else "testing")[0]) is not None:
        done = False
        previous_state_index = None
        previous_action = None
        previous_reward = 0
        total_episode_reward = 0
        total_drawdown = 0
        # Run daily data
        while not done:
            # Get action according to the model
            if previous_action is None:
                previous_buy = model(torch.tensor(current_state + [1]))
                previous_sell = model(torch.tensor(current_state + [0]))
                action = int(torch.argmax((previous_buy + previous_sell)))
            else:
                action = int(torch.argmax(model(torch.tensor(current_state + [previous_action]))))
            # Execute the action
            current_state, _, previous_reward, done = env.step(action)
            previous_action = action
            total_episode_reward += previous_reward
            total_drawdown = min(total_drawdown, total_episode_reward)
        rewards_per_episode.append(total_episode_reward)
        drawdown_per_episode.append(total_drawdown)
    return rewards_per_episode, drawdown_per_episode


def run(data_path, parameters={"LEARNING_RATE" : conf.RL_LEARNING_RATE,
                               "DISCOUNTED_FUTURE_REWARD_FACTOR": conf.RL_DISCOUNTED_FUTURE_REWARD_FACTOR,
                               "LAYER_SIZES": conf.DQN_LAYER_SIZES,
                               "TARGET_UPDATE_FREQUENCY": conf.DQN_TARGET_UPDATE_FREQUENCY,
                               "BATCH_SIZE": conf.DQN_BATCH_SIZE,
                               "TARGET_UPDATE_WEIGHT": conf.DDQN_TARGET_UPDATE_WEIGHT
                              },
        save_model=True, verbose=True):
    """
    Runs the DDQN algorithm.
    
    Args:
            data_path: path to the OFI data
            parameters: hyperparameters for the model
            save_model: saves the DDQN policy model to disk
            verbose: prints updates to the command line
    Returns:
            the total reward per episode, the maximum drawdown per episode for
            both validation and testing datasets
    """
    # Create models
    if verbose:
        print("Creating models")
    instrument = data_path.split('/')[-1].split('.')[0]
    
    POLICY_NET = DeepQNetwork(parameters["LAYER_SIZES"])
    TARGET_NET = DeepQNetwork(parameters["LAYER_SIZES"])

    TARGET_NET.load_state_dict(POLICY_NET.state_dict())
    TARGET_NET.eval()
    buffer = deque([], conf.DQN_BUFFER_SIZE)
    optimizer = optim.Adam(POLICY_NET.parameters(), lr=parameters["LEARNING_RATE"], weight_decay=conf.L2_REGULARISATION)
    # Load data
    if verbose:
        print("Loading data")
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
        current_state = env.next_ep(dataset="training")[0]
        done = False
        previous_state = None
        previous_action_2 = None
        previous_action = None
        previous_reward = 0
        step_count = 0
        # Run daily data
        while not done:
            # Make random action if need to explore
            if np.random.uniform(0, 1) < exploration_rate:
                action = env.action_space_sample()
            else:
                # Get action according to the model
                if previous_action is None:
                    previous_buy = POLICY_NET(torch.tensor(current_state + [1]))
                    previous_sell = POLICY_NET(torch.tensor(current_state + [0]))
                    action = int(torch.argmax((previous_buy + previous_sell)))
                else:
                    action = int(torch.argmax(POLICY_NET(torch.tensor(current_state +
                                                                      [previous_action]))))
            # Add data to the replay experience buffer
            if previous_action_2 != None:
                buffer.append([torch.tensor(previous_state + [previous_action_2]),
                               torch.tensor([previous_action]),
                               torch.tensor(current_state + [previous_action]),
                               torch.tensor([previous_reward])])
            # Update metrics for next iteration
            previous_state = current_state
            previous_action_2 = previous_action
            previous_action = action
            # Execute the action
            current_state, _, previous_reward, done = env.step(action)
            total_episode_reward += previous_reward
            total_drawdown = min(total_drawdown, total_episode_reward)
            if not len(buffer) < conf.DQN_BATCH_SIZE and step_count % 10 == 0:
                transitions = random.sample(buffer, conf.DQN_BATCH_SIZE)
                state_batch, action_batch, nextstate_batch, reward_batch = (torch.stack(x) \
                                                                            for x in zip(*transitions))
                # Compute the loss
                next_actions_using_policy_dqn = torch.argmax(POLICY_NET(nextstate_batch),
                                                             dim=1).unsqueeze(1)
                next_q_values_using_target_dqn = TARGET_NET(nextstate_batch) \
                    .gather(1, next_actions_using_policy_dqn).squeeze()
                bellman_targets = parameters["DISCOUNTED_FUTURE_REWARD_FACTOR"][instrument] * \
                    next_q_values_using_target_dqn + reward_batch.reshape(-1) 
                q_values = POLICY_NET(state_batch).gather(1, action_batch).reshape(-1)
                mse_loss = ((q_values - bellman_targets)**2).mean()
                # Optimize the model
                optimizer.zero_grad()
                mse_loss.backward()
                optimizer.step()
            # Update the target network
            if step_count % parameters["TARGET_UPDATE_FREQUENCY"][instrument] == 0:
                update_target_network(POLICY_NET, TARGET_NET,
                                      parameters["TARGET_UPDATE_WEIGHT"][instrument])
            step_count += 1
        if verbose:
            print(f"Epoch: {episode + 1}\t\tTraining Total Reward: {total_episode_reward}")
        # Update exploration rate
        exploration_rate = max(conf.RL_MIN_EXPLORATION_RATE,
                               exploration_rate * conf.RL_EXPLORATION_DECAY)
        rewards_per_episode.append(total_episode_reward)
        drawdown_per_episode.append(total_drawdown)
    # Validation
    if verbose:
        print("Validation")
    rewards_per_episode_v, drawdown_per_episode_v = test_run(env, POLICY_NET, validation=True)
    if verbose:
        print(f"Validation Rewards: {rewards_per_episode_v}")
        print(f"Validation Drawdown: {drawdown_per_episode_v}")
    # Testing
    if verbose:
        print("Testing")
    rewards_per_episode_t, drawdown_per_episode_t = test_run(env, POLICY_NET, validation=False)
    if verbose:
        print(f"Testing Rewards: {rewards_per_episode_t}")
        print(f"Testing Drawdown: {drawdown_per_episode_t}")
    # Save the model
    if save_model:
        torch.save(POLICY_NET.state_dict(), '../models/dqn/' + instrument + '.pt')
    return rewards_per_episode_v, drawdown_per_episode_v, rewards_per_episode_t, drawdown_per_episode_t
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m trainers.ddqn path_to_ofi_data")
        sys.exit(0)
    run(sys.argv[1])

