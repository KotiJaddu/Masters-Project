""""""""""""""""""""
#  Contains utility methods, model architectures, and a reinforcement learning
#  environment to allow iterating through historical data.
""""""""""""""""""""
from datetime import datetime

import numpy as np
import pandas as pd
import torch

import config as conf


def get_data_and_alphas(data_path):
    """
    Uses raw collected data and splits it into training, validation, and testing data.
    
    Args:
            data_path: path to the raw collected data file containing OFI features
    Returns:
            training, validation, and testing datasets along with dates
            contained in each respective dataset
    """
    df = pd.read_csv(data_path)
    # Extract the dates from the time column in the data
    df['Date'] = df['Time'].str.split(' ').str[0].str.strip()
    unique_dates = list(df['Date'].unique())
    list_df = []
    # Iterate through unique dates
    for date in unique_dates:
        df_date = df[df['Date'] == date].reset_index(drop=True)
        # Calculate alphas
        for i in range(1, 1 + conf.HORIZON):
            df_date[f'Horizon {i}'] = df_date.shift(periods=-i)[['Mid Price']]
            df_date[f'Alpha {i}'] = (df_date[f'Horizon {i}'] - df_date['Mid Price'])
            df_date = df_date.drop(f'Horizon {i}', axis=1)
        df_date = df_date.iloc[:-conf.HORIZON,:]
        # Add daily data to variable list_df
        list_df.append(df_date)
    # Concatenate daily data
    df = pd.concat(list_df).reset_index(drop=True)
    unique_dates = list(df['Date'].unique())
    unique_dates_sorted = sorted(unique_dates, key=lambda x: datetime.strptime(x, '%d/%m/%Y'))
    
    validation_start = int(conf.VALIDATION_START * len(unique_dates))
    test_start = int(conf.TEST_START * len(unique_dates))
    
    # Split daily data into training, validation, and testing datasets
    training_dates = unique_dates_sorted[:validation_start]
    validation_dates = unique_dates_sorted[validation_start:test_start]
    testing_dates = unique_dates_sorted[test_start:]
    
    training_data = df[df['Date'].isin(training_dates)].reset_index(drop=True)
    validation_data = df[df['Date'].isin(validation_dates)].reset_index(drop=True)
    testing_data = df[df['Date'].isin(testing_dates)].reset_index(drop=True)
    
    return (training_data, validation_data, testing_data,
            training_dates, validation_dates, testing_dates)


class RegressionNetwork(torch.nn.Module):
    """Class that describes the regression network for alpha extraction."""
    
    def __init__(self, layer_sizes):
        """
        Initialises the regression network.

        Args:
                layer_sizes: list with size of each layer as elements
        """
        super().__init__()
        # Create a ModuleList of all the layers in the network
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(conf.ORDER_BOOK_STORAGE_LIMIT, layer_sizes[0])] +
            [torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)] + 
            [torch.nn.Linear(layer_sizes[-1], conf.HORIZON)]
        )
                
    def forward(self, x):
        """
        Performs the forward pass of the network.

        Args:
                x: an input tensor (OFI data in this case)
        Returns:
                alpha predictions
        """
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)


class DeepQNetwork(torch.nn.Module):
    """Class that describes the deep Q network for trade signals."""
    
    def __init__(self, layer_sizes):
        """
        Initialises the DQN.

        Args:
            layer_sizes: list with size of each layer as elements
        """
        super(DeepQNetwork, self).__init__()
        # Create a ModuleList of all the layers in the network
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(conf.HORIZON + 1, layer_sizes[0])] +
            [torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)] + 
            [torch.nn.Linear(layer_sizes[-1], conf.ACTION_SPACE)]
        )
    
    def forward (self, x):
        """
        Performs the forward pass of the DQN.

        Args:
                x: an input tensor (alphas and the previous action in this case)
        Returns:
                the next action
        """
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = torch.sigmoid(self.layers[-1](x))
        return x

    
class Environment:
    """Class that describes the environment for rl agents to iterate through historical data."""
    
    def __init__(self, data_path):
        """
        Initialises the environment.

        Args:
                data_path: path to the raw collected data file containing OFI features
        """
        # Initialise attributes
        self.training_data, self.validation_data, self.testing_data, \
        self.training_dates, self.validation_dates, self.testing_dates = get_data_and_alphas(data_path)
        self.instrument = data_path.split('/')[-1].split('.')[0]
        self.spread = conf.SPREAD[self.instrument]
        self.previous_action = None
        self.previous_price = None
        self.current_day_tick = 0
        self.current_day = None
        self.current_day_data = None
        self.episode = 0
        self.validation_day_count = 0
        self.testing_day_count = 0
    
    def next_ep(self, dataset="training"):
        """
        Loads the next episode (next day) within different datasets.

        Args:
                dataset: string which describes whether to use the training, validation, or
                testing dataset
            
        Returns:
                the next state as alphas, the next state as OFI values, the previous reward,
                and whether the episode (trading day) is done
        """
        # Get the next episode from the training dataset
        if dataset=="training":
            self.current_day = self.training_dates[self.episode % len(self.training_dates)]
            self.current_day_data = self.training_data[self.training_data['Date'] == \
                                                       self.current_day].reset_index(drop=True)
        # Get the next episode from the validation dataset
        elif dataset=="validation":
            if self.validation_day_count == len(self.validation_dates):
                self.validation_day_count = 0
                return None, None
            self.current_day = self.validation_dates[self.validation_day_count]
            self.validation_day_count += 1
            self.current_day_data = self.validation_data[self.validation_data['Date'] == \
                                                         self.current_day].reset_index(drop=True)
        # Get the next episode from the testing dataset
        elif dataset=="testing":
            if self.testing_day_count == len(self.testing_dates):
                self.testing_day_count = 0
                return None, None
            self.current_day = self.testing_dates[self.testing_day_count]
            self.testing_day_count += 1
            self.current_day_data = self.testing_data[self.testing_data['Date'] == \
                                                      self.current_day].reset_index(drop=True)
        
        self.current_day_alphas = torch.Tensor(self.current_day_data.drop(['Time', 'OFI', 'Mid Price', 'Date'],
                                                                          axis=1).values)
        # Reset metrics
        self.current_day_tick = 0
        self.previous_action = None
        self.previous_price = None
        
        # Calculate next states
        next_state_ofi = torch.FloatTensor([float(x) for x in \
                                            self.current_day_data \
                                            .iloc[self.current_day_tick]["OFI"][1:-1].split(',')])
        next_state_alphas = self.current_day_alphas[self.current_day_tick].tolist()
        self.episode += 1
        return next_state_alphas, next_state_ofi
    
    def step(self, action):
        """
        Applies a step in the environment and returns the next states.

        Args:
                action: an action describing whether to buy or sell in the environment
            
        Returns:
                the next state as alphas, the next state as OFI values, the previous reward,
                and whether the episode (trading day) is done
        """
        
        previous_reward = 0
        # If the action is applicable, apply it to the environment
        if action is not None:
            current_price = self.current_day_data.iloc[self.current_day_tick]["Mid Price"]
            
            if self.previous_action is not None:
                # Calculate the previous reward after observing a change in price
                previous_reward = (current_price - self.previous_price) * \
                    (self.previous_action * 2 - 1) - (self.spread if self.add_spread else 0)
            self.add_spread = self.previous_action != action

            self.previous_price = current_price
            self.previous_action = action
        
        # Calculate the next states
        self.current_day_tick += 1
        done = len(self.current_day_alphas) - 1 <= self.current_day_tick
        next_state_alphas, next_state_ofi  = None, None
        if len(self.current_day_alphas) - 1 > self.current_day_tick:
            next_state_ofi = torch.FloatTensor([float(x) for x in \
                                                self.current_day_data \
                                                .iloc[self.current_day_tick]["OFI"][1:-1].split(',')])
            next_state_alphas = self.current_day_alphas[self.current_day_tick].tolist()
        
        return next_state_alphas, next_state_ofi, previous_reward, done
    
  
    
    def action_space_sample(self):
        """
        Sample a random action from the action space.
            
        Returns:
                a random action in the action space
        """
        if np.random.uniform(0, 1) < 0.5:
            return 1
        return 0
