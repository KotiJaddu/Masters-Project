""""""""""""""""""""
#  Hosts a web server and parses OFI data from the CTrader FXPro platform
#  script (through POST requests) and sends signals for making trades on the
#  platform using the best performing reinforcement agent for the best instrument.
""""""""""""""""""""
import sys

from flask import Flask
import numpy as np
import torch

import config as conf
from trainers.qLearning import get_state_in_Q_table_index
from utils import DeepQNetwork, RegressionNetwork


PREVIOUS_ACTION = None

app = Flask(__name__)


@app.route('/<ofi>')
def get_action(ofi):
    """
    Awaits for requests and parses incoming OFI data, passes it through the machine learning
    pipelines to return the optimal action back to the caller.
    
    Args:
            ofi: the OFI data received from CTrader as a string
    Returns:
            the optimal action to execute on CTrader
    """
    global PREVIOUS_ACTION
    # Parse OFI data
    ofi_tensor = torch.Tensor([float(x) for x in ofi.split('=')[1].split('_')])
    # Extract alphas using alpha extractor
    alphas = alpha_extractor(ofi_tensor).detach().numpy()
    # If using Q Learning
    if rl_agent_name == "qLearning":
        min_state_value_with_data = conf.MIN_STATE_VALUE[conf.INSTRUMENT_FORWARD_TESTING]
        max_state_value_with_data = conf.MAX_STATE_VALUE[conf.INSTRUMENT_FORWARD_TESTING]
        # Get state index
        state_index = get_state_in_Q_table_index(alphas, min_state_value_with_data,
                                                 max_state_value_with_data)
        # If state index is out of range, make no action
        if min(state_index) < 0 or max(state_index) >= rl_agent.shape[0]:
            action = -1
        else:
            # Get action according to the model
            if PREVIOUS_ACTION != None:
                action = np.argmax(rl_agent[state_index + (PREVIOUS_ACTION,)])
            else: 
                previous_buy = rl_agent[state_index + (1,)]
                previous_sell = rl_agent[state_index + (0,)]
                action = int(np.argmax((previous_buy + previous_sell)))
    # If using DQN or DDQN (does not matter as process is the same)
    else:
        # Get action according to the model
        if PREVIOUS_ACTION != None:
            action = np.argmax(rl_agent(torch.Tensor(alphas.tolist() +
                                                     [PREVIOUS_ACTION])).detach().numpy())
        else:
            previous_buy = rl_agent(torch.tensor(alphas.tolist() + [1]))
            previous_sell = rl_agent(torch.tensor(alphas.tolist() + [0]))
            action = int(torch.argmax((previous_buy + previous_sell)))
    # Update previous action and return the new action
    PREVIOUS_ACTION = action
    return str(action)

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1].lower() not in ["qlearning", "dqn", "ddqn"]:
        print("usage: python -m forwardTestSignals qLearning/dqn/ddqn")
        sys.exit(0)
    # Load alpha extractor
    alpha_extractor = RegressionNetwork(conf.SL_LAYER_SIZES)
    alpha_extractor.load_state_dict(torch.load('../models/alpha_extraction/' +
                                               conf.FORWARD_TESTING_INSTRUMENT + '.pt',
                                               map_location=torch.device('cpu')))
    alpha_extractor.eval()
    
    # Parse RL agent to use
    rl_agent_name = sys.argv[1].lower()
    # Load Q Learning table
    if rl_agent_name == "qlearning":
        rl_agent = torch.Tensor(torch.load('../models/q_learning/'+ conf.FORWARD_TESTING_INSTRUMENT + \
                                           '.pt'))
    # Load DQN model
    elif rl_agent_name == "dqn":
        rl_agent = DeepQNetwork(conf.DQN_LAYER_SIZES)
        rl_agent.load_state_dict(torch.load('../models/dqn/'+ conf.FORWARD_TESTING_INSTRUMENT + '.pt',
                                            map_location=torch.device('cpu')))
        rl_agent.eval()
    # Load DDQN model
    elif rl_agent_name == "ddqn":
        rl_agent = DeepQNetwork(conf.DQN_LAYER_SIZES)
        rl_agent.load_state_dict(torch.load('../models/ddqn/'+ conf.FORWARD_TESTING_INSTRUMENT + '.pt',
                                            map_location=torch.device('cpu')))
        rl_agent.eval() 
    app.run()
