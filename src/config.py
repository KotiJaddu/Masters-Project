""""""""""""""""""""
#  Contains global variables used in this project.
""""""""""""""""""""
import numpy as np
import torch

### GENERAL PROJECT METRICS

HORIZON = 6
ORDER_BOOK_STORAGE_LIMIT = 10
ACTION_SPACE = 2
VALIDATION_START = 0.8
TEST_START = 0.9

L2_REGULARISATION = 1e-5
USE_GPU = True
DTYPE = torch.float32

### HYPERPARAMETERS FOR SUPERVISED LEARNING COMPONENT

SL_LAYER_SIZES = [2048] * 4

SL_LEARNING_RATE = {"XAUUSD": 0.0001,
                    "GBPUSD": 0.0001,
                    "EURUSD": 0.0001,
                    "#UK100": 0.00001,
                    "#Germany40": 0.00001
                   }

SL_EARLY_STOPPING = 5
SL_BATCH_SIZE = 256
SL_MAXIMUM_NUMBER_OF_EPOCHS = 100

### HYPERPARAMETERS FOR REINFORCEMENT LEARNING COMPONENT

RL_EXPLORATION_RATE = 1
RL_MIN_EXPLORATION_RATE = 0.01

RL_LEARNING_RATE = 0.01
RL_DISCOUNTED_FUTURE_REWARD_FACTOR = {"XAUUSD": 0.9,
                                      "GBPUSD": 0.95,
                                      "EURUSD": 0.95,
                                      "#UK100": 0.95,
                                      "#Germany40": 0.95
                                     }
RL_EXPLORATION_DECAY = 0.935
RL_NUMBER_OF_EPISODES = 80

### Q LEARNING

# During training
Q_MAX_STATE_VALUE_TRAINING = {"XAUUSD": 2.5 * np.array([0.0227, 0.0336, 0.0423, 0.0496, 0.0561, 0.0619]),
                              "GBPUSD": 2.5 * np.array([1.4650e-05, 2.1672e-05, 2.7318e-05, 3.2047e-05, 3.6275e-05,
                                                        4.0048e-05]),
                              "EURUSD": 2.5 * np.array([1.3003e-05, 1.8950e-05, 2.3782e-05, 2.7773e-05, 3.1335e-05,
                                                        3.4540e-05]),
                              "#UK100": 2.5 * np.array([0.3079, 0.4024, 0.4864, 0.5535, 0.6159, 0.6707]),
                              "#Germany40": 2.5 * np.array([0.9994, 1.0490, 1.3231, 1.4246, 1.6026, 1.7087])
                             }

Q_MIN_STATE_VALUE_TRAINING = {"XAUUSD": -2.5 * np.array([0.0227, 0.0336, 0.0423, 0.0496, 0.0561, 0.0619]),
                              "GBPUSD": -2.5 * np.array([1.4650e-05, 2.1672e-05, 2.7318e-05, 3.2047e-05, 3.6275e-05,
                                                         4.0048e-05]),
                              "EURUSD": -2.5 * np.array([1.3003e-05, 1.8950e-05, 2.3782e-05, 2.7773e-05, 3.1335e-05,
                                                         3.4540e-05]),
                              "#UK100": -2.5 * np.array([0.3079, 0.4024, 0.4864, 0.5535, 0.6159, 0.6707]),
                              "#Germany40": -2.5 * np.array([0.9994, 1.0490, 1.3231, 1.4246, 1.6026, 1.7087])
                             }

# During forward testing
Q_MAX_STATE_VALUE = {"XAUUSD": 2.5 * np.array([0.0236, 0.0348, 0.0438, 0.0515, 0.0581, 0.0641]),
                     "GBPUSD": 2.5 * np.array([1.5525e-05, 2.2957e-05, 2.8921e-05, 3.3897e-05, 3.8329e-05,
                                               4.2278e-05]),
                     "EURUSD": 2.5 * np.array([1.3775e-05, 2.0038e-05, 2.5104e-05, 2.9283e-05, 3.3004e-05,
                                               3.6348e-05]),
                     "#UK100": 2.5 * np.array([0.3096, 0.4082, 0.4945, 0.5638, 0.6278, 0.6841]),
                     "#Germany40": 2.5 * np.array([1.0063, 1.0717, 1.3456, 1.4572, 1.6361, 1.7496])
                    }

Q_MIN_STATE_VALUE = {"XAUUSD": -2.5 * np.array([0.0236, 0.0348, 0.0438, 0.0515, 0.0581, 0.0641]),
                     "GBPUSD": -2.5 * np.array([1.5525e-05, 2.2957e-05, 2.8921e-05, 3.3897e-05, 3.8329e-05,
                                                4.2278e-05]),
                     "EURUSD": -2.5 * np.array([1.3775e-05, 2.0038e-05, 2.5104e-05, 2.9283e-05, 3.3004e-05,
                                                3.6348e-05]),
                     "#UK100": -2.5 * np.array([0.3096, 0.4082, 0.4945, 0.5638, 0.6278, 0.6841]),
                     "#Germany40": -2.5 * np.array([1.0063, 1.0717, 1.3456, 1.4572, 1.6361, 1.7496])
                    }

Q_SIZE_PER_BUCKET = 5

### DQN / DDQN

DQN_LAYER_SIZES = [64] * 2
DQN_BUFFER_SIZE = 10_000
DQN_BATCH_SIZE = 128
DQN_TARGET_UPDATE_FREQUENCY = {"XAUUSD": 6,
                               "GBPUSD": 3,
                               "EURUSD": 3,
                               "#UK100": 3,
                               "#Germany40": 3
                              }
DDQN_TARGET_UPDATE_WEIGHT = {"XAUUSD": 0.1,
                             "GBPUSD": 0.2,
                             "EURUSD": 0.2,
                             "#UK100": 0.2,
                             "#Germany40": 0.2
                            }

### TRAINING ENVIRONMENT

SPREAD = {"XAUUSD" : 0.045, "GBPUSD" : 0.000006, "EURUSD" : 0.000006, "#Germany40" : 0.6, "#UK100" : 1.00}

### BACKTESTING

PRICE_TO_PNL = {"XAUUSD": 1/0.01,
                "GBPUSD": 1/0.00001,
                "EURUSD": 1/0.00001,
                "#UK100": 1, 
                "#Germany40": 1}

### FORWARD TESTING

FORWARD_TESTING_INSTRUMENT = "GBPUSD" 

### PLOTS

Q_VALUES_DQN_SHORT_TO_LONG = np.array([[-0.82207313, -0.29782856, -0.65124264,  0.15896963,  0.6326571 ],
                                       [-0.11648637,  0.16240743, -0.84789822,  0.17106132,  0.40704244],
                                       [ 0.27062684, -0.71867126, -0.67093573, -0.64795517,  0.33396173],
                                       [ 0.29036895,  0.16928108, -0.47466491, -0.0992689,   0.32525989],
                                       [ 0.16483727, -0.78139078, -0.65767649, -0.40565722,  0.36757729],
                                       [ 0.30556802, -0.6576908,  -0.72373973,  0.12909376,  0.59527991]])


Q_VALUES_DQN_LONG_TO_SHORT = np.array([[ 0.09438097, -0.20337381,  -0.63648275,  0.13508572, -0.58266453],
                                       [ 0.73186958, -0.57562549, -0.49033178,  0.02096341, -0.16667101],
                                       [ 0.88311849,  0.62123037, -0.2805245,  -0.2933561,   0.29259781],
                                       [ 0.7217868,   0.1941513,  -0.3319957,  -0.04195925,  0.19886166],
                                       [ 0.27675665,  0.09274459, -0.46727547,  0.19189516,  0.3918493 ],
                                       [-0.5580099,   -0.10155419,  -0.58566215,  0.2876832,  0.13946218]])

Q_VALUES_DDQN_SHORT_TO_LONG = np.array([[ -0.1202677,  -0.32415915, -0.67927325,  0.78833786,  0.7771043 ],
                                        [ -0.13316829, -0.0575538,  -0.47767178, -0.16540716,  0.3133153 ],
                                        [ 0.07343747, -0.30353962, -0.45610371,  0.45840567,  0.08516143],
                                        [ 0.24165948, 0.2178897,  -0.55258285,  0.19853351,  0.60304772],
                                        [ 0.16842883, 0.2841595,  -0.61081946,  0.14250019,  0.76017727],
                                        [ 0.12235333, 0.15212038, -0.53751363, -0.21926478,  0.57662229]])

Q_VALUES_DDQN_LONG_TO_SHORT = np.array([[ 0.36917155,  0.7464633,  -0.65613719, -0.36964178,  -0.1667287 ],
                                        [ 0.49744645, -0.46072251, -0.61226314, -0.60924694, 0.14358346],
                                        [ 0.4310914,   0.34558177, -0.32075509, -0.72379832, 0.14193969],
                                        [ 0.38015833,  0.14591408, -0.48352432, -0.50956553,  0.06250777],
                                        [ 0.2477754,  -0.02643792, -0.30621537, 0.02217805,  -0.44824914],
                                        [ 0.32863022, -0.21808046, -0.50635276, 0.14076639,  -0.38835752]])
