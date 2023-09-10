""""""""""""""""""""
#  Runs the backtesting combining the alpha extraction with each reinforcement learning agent
#  on test data and returns performance metrics for each instrument.
""""""""""""""""""""
import sys

import numpy as np
import torch

import config as conf
from trainers.qLearning import get_state_in_Q_table_index
from utils import DeepQNetwork, Environment, RegressionNetwork


def calculate_metrics(instrument, rewards_per_episode, reward_per_trade):
    """
    Calculates performance metrics and returns them.
    
    Args:
            instrument: a string containing the instrument name
            rewards_per_episode: the reward gained per episode during testing (day)
            reward_per_trade: the reward gained after each trade during testing
    Returns:
            the performance metrics: daily average return, daily average volatility,
                average profit/loss ratio, and % profitability
    """
    
    reward_per_trade_np = np.array(reward_per_trade)
    
    daily_average_return = np.mean(np.array(rewards_per_episode) * conf.PRICE_TO_PNL[instrument])
    
    daily_average_volatility = np.std(np.array(rewards_per_episode) * conf.PRICE_TO_PNL[instrument])
    
    average_profit_loss = np.mean(reward_per_trade_np[reward_per_trade_np > 0]) / \
        np.mean(reward_per_trade_np[reward_per_trade_np < 0])
    
    profitability = 100 * len(reward_per_trade_np[reward_per_trade_np > 0]) / \
        (len(reward_per_trade_np[reward_per_trade_np > 0]) + 
         len(reward_per_trade_np[reward_per_trade_np < 0]))

    return daily_average_return, daily_average_volatility, np.abs(average_profit_loss), profitability


def q_tableRunner(env, alpha_extractor, instrument):
    """
    Runs the backtesting script for the Q Learning algorithm and returns raw metrics.
    
    Args:
            env: the backtesting environment
            alpha_extractor: the model that will extract alphas when given OFI data
            instrument: a string containing the instrument name
    """
    # Load table
    q_table = torch.Tensor(torch.load('../models/q_learning/'+ instrument + '.pt'))
    min_state_value_with_data = conf.Q_MIN_STATE_VALUE_TRAINING[instrument]
    max_state_value_with_data = conf.Q_MAX_STATE_VALUE_TRAINING[instrument]
    reward_per_trade = []
    rewards_per_episode = []
    max_drawdown = 0
    equity = 0
    # Iterate through test dataset
    while (next_state_ofi := env.next_ep("testing")[1]) is not None:
        done = False
        previous_state_index = None
        previous_action = None
        previous_reward = 0
        total_episode_reward = 0
        pnl = 0
        # Run daily data
        while not done:
            # Forecast alphas using the alpha extractor
            next_state = alpha_extractor(next_state_ofi).detach().numpy()
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
            # Execute the action
            _, next_state_ofi, previous_reward, done = env.step(action)
            
            # Keep track of the results
            if previous_action != None and previous_action != action:
                reward_per_trade.append(pnl)
                pnl = 0
            previous_action = action
            pnl += float(previous_reward)
            total_episode_reward += float(previous_reward)
            equity += float(previous_reward)
            max_drawdown = min(max_drawdown, equity)
        rewards_per_episode.append(total_episode_reward)
    return rewards_per_episode, reward_per_trade, max_drawdown * conf.PRICE_TO_PNL[instrument]


def dqnRunner(env, alpha_extractor, instrument, ddqn=False):
    """
    Runs the backtesting script for the DQN and DDQN algorithm and returns raw metrics.
    
    Args:
            env: the backtesting environment
            alpha_extractor: the model that will extract alphas when given OFI data
            instrument: a string containing the instrument name
            ddqn: a boolean to use the DDQN algorithm
    """
    # Load model
    model = DeepQNetwork(layer_sizes=conf.DQN_LAYER_SIZES)
    model.load_state_dict(torch.load('../models/' + ('d' if ddqn else '') +
                                     'dqn/'+ instrument + '.pt', map_location=torch.device('cpu')))
    model.eval()
    reward_per_trade = []
    rewards_per_episode = []
    max_drawdown = 0
    equity = 0
    # Iterate through test dataset
    while (next_state_ofi := env.next_ep("testing")[1]) is not None:
        done = False
        previous_state_index = None
        previous_action = None
        previous_reward = 0
        total_episode_reward = 0
        pnl = 0
        # Run daily data
        while not done:
            # Forecast alphas using the alpha extractor
            next_state = alpha_extractor(next_state_ofi).detach().numpy().tolist()
            # Get action according to the model
            if previous_action is None:
                previous_buy = model(torch.tensor(next_state + [1]))
                previous_sell = model(torch.tensor(next_state + [0]))
                action = int(torch.argmax((previous_buy + previous_sell)))
            else:
                action = int(torch.argmax(model(torch.tensor(next_state + [previous_action]))))
            # Execute the action
            _, next_state_ofi, previous_reward, done = env.step(action)
            
            # Keep track of the results
            if previous_action != None and previous_action != action:
                reward_per_trade.append(pnl)
                pnl = 0
            previous_action = action
            pnl += float(previous_reward)
            total_episode_reward += float(previous_reward)
            equity += float(previous_reward)
            max_drawdown = min(max_drawdown, equity)
        rewards_per_episode.append(total_episode_reward)
    return rewards_per_episode, reward_per_trade, max_drawdown * conf.PRICE_TO_PNL[instrument]


def run(data_path):
    """
    Runs the backtesting script for each agent and prints the performance metrics.
    
    Args:
            data_path: the path to the OFI data
    """
    instrument = data_path.split('/')[-1].split('.')[0]
    alpha_extractor = RegressionNetwork(conf.SL_LAYER_SIZES)
    alpha_extractor.load_state_dict(torch.load('../models/alpha_extraction/' +
                                               instrument + '.pt'))
    alpha_extractor.eval()
    env = Environment(data_path)
    
    # Perform backtesting for Q Learning
    rewards_per_episode, reward_per_trade, max_drawdown = q_tableRunner(env, alpha_extractor, instrument)
    daily_average_return, daily_average_volatility, average_profit_loss, profitability = \
        calculate_metrics(instrument, rewards_per_episode, reward_per_trade)
    print(f"Q Learning for {instrument}")
    print({"daily_average_return": daily_average_return,
           "daily_average_volatility": daily_average_volatility,
           "average_profit_loss": average_profit_loss,
           "profitability": profitability,
           "max_drawdown": max_drawdown})
    
    # Perform backtesting for DQN
    rewards_per_episode, reward_per_trade, max_drawdown = dqnRunner(env, alpha_extractor, instrument, ddqn=False)
    daily_average_return, daily_average_volatility, average_profit_loss, profitability = \
        calculate_metrics(instrument, rewards_per_episode, reward_per_trade)
    print(f"DQN for {instrument}")
    print({"daily_average_return": daily_average_return,
           "daily_average_volatility": daily_average_volatility,
           "average_profit_loss": average_profit_loss,
           "profitability": profitability,
           "max_drawdown": max_drawdown})
    
    # Perform backtesting for DDQN
    rewards_per_episode, reward_per_trade, max_drawdown = dqnRunner(env, alpha_extractor, instrument, ddqn=True)
    daily_average_return, daily_average_volatility, average_profit_loss, profitability = \
        calculate_metrics(instrument, rewards_per_episode, reward_per_trade)
    print(f"DDQN for {instrument}")
    print({"daily_average_return": daily_average_return,
           "daily_average_volatility": daily_average_volatility,
           "average_profit_loss": average_profit_loss,
           "profitability": profitability,
           "max_drawdown": max_drawdown})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m backtesting path_to_data_dir")
        sys.exit(0)
    for instrument in ["#UK100", "#Germany40", "XAUUSD", "GBPUSD", "EURUSD"]: 
        print(f"Running backtest for {instrument}")
        run(sys.argv[1]  + instrument + ".csv")
