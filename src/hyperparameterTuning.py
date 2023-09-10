""""""""""""""""""""
#  Finds and returns the best configuration of hyperparameters for all
#  models for each instrument.
""""""""""""""""""""
import sys

import numpy as np

from trainers.alphaExtraction import run as alphaExtractionRunner
from trainers.dqn import run as dqnRunner
from trainers.ddqn import run as ddqnRunner
from trainers.qLearning import run as qLearningRunner

INSTRUMENTS =  ["XAUUSD", "GBPUSD", "EURUSD", "#UK100", "#Germany40"]


def tune_alphaExtraction(data_directory):
    """
    Tunes the hyperparameters for the alpha extraction.
    
    Args:
            data_directory: the directory of the data
    """
    hyperparameter_search_results = []
    # Configurations
    hyperparametersearch = {"LAYER_SIZES": [[512] * 4, [1024] * 4, [2048] * 4],
                            "LEARNING_RATE": [0.00001, 0.0001],
                            "EARLY_STOPPING": [5, 10],
                            "BATCH_SIZE": [128, 256, 512]
                           }
    params = []
    total_steps = np.prod([len(trials) for trials in hyperparametersearch.values()])
    scores = []
    for instrument in INSTRUMENTS:
        step = 0
        # Iterate through all configurations
        for layer_sizes in hyperparametersearch["LAYER_SIZES"]:
            for learning_rate in hyperparametersearch["LEARNING_RATE"]:
                for early_stopping in hyperparametersearch["EARLY_STOPPING"]:
                    for batch_size in hyperparametersearch["BATCH_SIZE"]:
                        current_parameters = {"LAYER_SIZES": layer_sizes,
                                              "LEARNING_RATE": learning_rate,
                                              "EARLY_STOPPING": early_stopping,
                                              "BATCH_SIZE": batch_size
                                             }
                        params.append(str(current_parameters))
                        print(f"\nStep {step+1}/{total_steps} for {instrument} : {current_parameters}\n")
                        step += 1
                        # Find validation loss
                        average_validation_score = \
                            alphaExtractionRunner(data_path=data_directory + instrument + ".csv",
                                                  parameters=current_parameters,
                                                  save_model=False, verbose=False)
                        scores.append(average_validation_score)
                        print(f"Validation Error: {average_validation_score:.12f}")
        # Get the best parameters that gave the smallest validation loss
        # for each instrument and print them
        best_parameters = params[scores.index(min(scores))]
        print(f"Best Hyperparameters for Alpha Extraction for {instrument}: " + \
                best_parameters + "\n")
        hyperparameter_search_results.append(f"{instrument}: {best_parameters}")
    # Print the best parameters for all instruments
    print(f"Final Results: {hyperparameter_search_results}")

        
def tune_qLearning(data_directory):
    """
    Tunes the hyperparameters for the Q Learning algorithm.
    
    Args:
            data_directory: the directory of the data
    """
    hyperparameter_search_results = []
    # Configurations
    hyperparametersearch = {"LEARNING_RATE" : [0.01, 0.001],
                            "DISCOUNTED_FUTURE_REWARD_FACTOR": [0.9, 0.95]
                           }
    params = []
    total_steps = np.prod([len(trials) for trials in hyperparametersearch.values()])
    scores = []
    for instrument in INSTRUMENTS:
        step = 0
        # Iterate through all configurations
        for learning_rate in hyperparametersearch["LEARNING_RATE"]:
            for discounted_future_reward_factor in hyperparametersearch["DISCOUNTED_FUTURE_REWARD_FACTOR"]:
                current_parameters = {"LEARNING_RATE": learning_rate,
                                      "DISCOUNTED_FUTURE_REWARD_FACTOR": discounted_future_reward_factor
                                     }
                params.append(str(current_parameters))
                print(f"\nStep {step+1}/{total_steps} for {instrument} : {current_parameters}\n")
                step += 1
                # Find validation loss
                validation_score = \
                    qLearningRunner(data_path=data_directory + instrument + ".csv",
                                    parameters=current_parameters,
                                    save_model=False, verbose=False)[0]
                average_validation_score = np.mean(np.array(validation_score))
                scores.append(average_validation_score)
                print(f"Validation Score: {average_validation_score:.12f}")
        # Get the best parameters that gave the smallest validation loss
        # for each instrument and print them
        best_parameters = params[scores.index(max(scores))]
        print(f"Best Hyperparameters for Tabular Q Learning for {instrument}: " + \
                best_parameters + "\n")
        hyperparameter_search_results.append(f"{instrument}: {best_parameters}")
    # Print the best parameters for all instruments
    print(f"Final Results: {hyperparameter_search_results}")

    
def tune_dqn(data_directory):
    """
    Tunes the hyperparameters for the DQN algorithm.
    
    Args:
            data_directory: the directory of the data
    """
    hyperparameter_search_results = []
    # Configurations
    hyperparametersearch = {"LEARNING_RATE" : [0.01, 0.001],
                            "DISCOUNTED_FUTURE_REWARD_FACTOR": [0.9, 0.95],
                            "LAYER_SIZES": [[32] * 2, [64] * 2],
                            "TARGET_UPDATE_FREQUENCY": [3, 6],
                            "BATCH_SIZE": [128, 256]
                           }
    params = []
    total_steps = np.prod([len(trials) for trials in hyperparametersearch.values()])
    scores = []
    for instrument in INSTRUMENTS:
        step = 0
        # Iterate through all configurations
        for learning_rate in hyperparametersearch["LEARNING_RATE"]:
            for discounted_future_reward_factor in hyperparametersearch["DISCOUNTED_FUTURE_REWARD_FACTOR"]:
                for layer_sizes in hyperparametersearch["LAYER_SIZES"]:
                    for target_update_frequency in hyperparametersearch["TARGET_UPDATE_FREQUENCY"]:
                        for batch_size in hyperparametersearch["BATCH_SIZE"]:
                            current_parameters = {"LEARNING_RATE" : learning_rate,
                                                  "DISCOUNTED_FUTURE_REWARD_FACTOR":
                                                   discounted_future_reward_factor,
                                                  "LAYER_SIZES": layer_sizes,
                                                  "TARGET_UPDATE_FREQUENCY": target_update_frequency,
                                                  "BATCH_SIZE": batch_size
                                                 }
                            params.append(str(current_parameters))
                            print(f"\nStep {step+1}/{total_steps} for {instrument} : {current_parameters}\n")
                            step += 1
                            # Find validation loss
                            validation_score = \
                                dqnRunner(data_path=data_directory + instrument + ".csv",
                                          parameters=current_parameters,
                                          save_model=False, verbose=False)[0]
                            average_validation_score = np.mean(np.array(validation_score))
                            scores.append(average_validation_score)
                            print(f"Validation Score: {average_validation_score:.12f}")
        # Get the best parameters that gave the smallest validation loss
        # for each instrument and print them
        best_parameters = params[scores.index(min(scores))]
        print(f"Best Hyperparameters for Deep Q Network for {instrument}: " + \
                best_parameters + "\n")
        hyperparameter_search_results.append(f"{instrument}: {best_parameters}")
    # Print the best parameters for all instruments
    print(f"Final Results: {hyperparameter_search_results}")
    
    
def tune_ddqn(data_directory):
    """
    Tunes the hyperparameters for the DDQN algorithm.
    
    Args:
            data_directory: the directory of the data
    """
    hyperparameter_search_results = []
    # Configurations
    hyperparametersearch = {"LEARNING_RATE" : [0.01, 0.001],
                            "DISCOUNTED_FUTURE_REWARD_FACTOR": [0.9, 0.95],
                            "LAYER_SIZES": [[32] * 2, [64] * 2],
                            "TARGET_UPDATE_FREQUENCY": [3, 6],
                            "BATCH_SIZE": [128, 256],
                            "TARGET_UPDATE_WEIGHT": [0.1, 0.2]
                           }
    params = []
    total_steps = np.prod([len(trials) for trials in hyperparametersearch.values()])
    scores = []
    for instrument in INSTRUMENTS:
        step = 0
        # Iterate through all configurations
        for learning_rate in hyperparametersearch["LEARNING_RATE"]:
            for discounted_future_reward_factor in hyperparametersearch["DISCOUNTED_FUTURE_REWARD_FACTOR"]:
                for layer_sizes in hyperparametersearch["LAYER_SIZES"]:
                    for target_update_frequency in hyperparametersearch["TARGET_UPDATE_FREQUENCY"]:
                        for batch_size in hyperparametersearch["BATCH_SIZE"]:
                            for target_update_weight in hyperparametersearch["TARGET_UPDATE_WEIGHT"]:
                                current_parameters = {"LEARNING_RATE" : learning_rate,
                                                      "DISCOUNTED_FUTURE_REWARD_FACTOR":
                                                       discounted_future_reward_factor,
                                                      "LAYER_SIZES": layer_sizes,
                                                      "TARGET_UPDATE_FREQUENCY": target_update_frequency,
                                                      "BATCH_SIZE": batch_size,
                                                      "TARGET_UPDATE_WEIGHT": target_update_weight
                                                     }

                                params.append(str(current_parameters))
                                print(f"\nStep {step+1}/{total_steps} for {instrument} : {current_parameters}\n")
                                step += 1
                                # Find validation loss
                                validation_score = \
                                    ddqnRunner(data_path=data_directory + instrument + ".csv",
                                               parameters=current_parameters,
                                               save_model=False, verbose=False)[0]
                                average_validation_score = np.mean(np.array(validation_score))
                                scores.append(average_validation_score)
                                print(f"Validation Score: {average_validation_score:.12f}")
        # Get the best parameters that gave the smallest validation loss
        # for each instrument and print them
        best_parameters = params[scores.index(min(scores))]
        print(f"Best Hyperparameters for Double Deep Q Network for {instrument}: " + \
                best_parameters + "\n")
        hyperparameter_search_results.append(f"{instrument}: {best_parameters}")
    # Print the best parameters for all instruments
    print(f"Final Results: {hyperparameter_search_results}")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m hyperparameterTuning path_to_data_dir")
        sys.exit(0)
    tune_alphaExtraction(sys.argv[1])
    tune_qLearning(sys.argv[1])
    tune_dqn(sys.argv[1])
    tune_ddqn(sys.argv[1])
