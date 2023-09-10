""""""""""""""""""""
#  Contains methods to generate plots and tables.
#
""""""""""""""""""""
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import torch

import config as conf
from trainers.alphaExtraction import convert_df_to_tensors
from trainers.qLearning import get_state_in_Q_table_index
from utils import get_data_and_alphas, RegressionNetwork


DATA_DIRECTORY = "/"


def create_std_and_mean_and_positive_negative_and_zero_proportions_alphas_tables():
    """Finds the standard deviation, mean, and positive, negative, and zero proportions in the alpha data."""
    nums = [12370183, 5498830, 3984114, 1270764, 1191069]
    nums = [523799, 358196, 272335, 187851, 171758]
    count = 0
    stds = []
    for instrument in ["XAUUSD", "GBPUSD", "EURUSD", "#UK100", "#Germany40"]:
        inputs = torch.FloatTensor(torch.load('../alphaInputs' + instrument + '.pt'))[-nums[count]:]
        count += 1
        mean = torch.mean(inputs, dim=0)
        std = torch.std(inputs, dim=0)
        val=[]
        for col in range(6):
            a = inputs[:,col: col+1].flatten()
            pos = len(a[a!=0])
            neg = len(a[a==0])
            val.append(pos*100/(pos+neg)) # Adjust for positive, negative, and zero proportions
        vals.append(val)
        mean = torch.mean(inputs, dim=0)
        std = torch.std(inputs, dim=0)
        means.append(mean.cpu().detach().numpy())
        stds.append(std.cpu().detach().numpy())
    print("vals")
    print(np.array(vals), 1)
    print(np.round(np.array(vals).T, 1))
    print("mean")
    print(np.array(means).T)
    print("std")
    print(np.array(stds).T)


def create_OFI_distribution_plots():
    """Creates the order flow imbalance histograms along with fitted normal distriubtion curve."""
    for instrument in ["EURUSD", "GBPUSD", "#Germany40", "#UK100", "XAUUSD"]:
        bins = 7
        inputs = torch.FloatTensor(torch.load('inputs' + instrument + '.pt'))
        mean = torch.mean(inputs, dim=0)
        std = torch.std(inputs, dim=0)
        fig, axs = plt.subplots(2, 5, constrained_layout = True)

        fig.suptitle(f"OFI Distributions for {instrument}", fontsize=16)
        for col in range(1, 11):
            x = np.linspace(min(inputs[:, col-1:col]), max(inputs[:, col-1:col]), bins).flatten()
            test = ((inputs - mean) / std)[:, col-1:col].flatten()
            hist = torch.histc(test, bins = bins, min = min(test), max = max(test))
            val = []
            for index, count in enumerate(hist.tolist()):
                val.extend([x[index]] * int(count))
            val = torch.FloatTensor(val)
            normal_hist = hist/sum(hist)
            axs[(col - 1) // 5, (col - 1) % 5].hist(inputs[:,col-1:col].flatten(), density=True,
                                                    bins=bins, histtype='step', fill=True)  
            axs[(col - 1) // 5, (col - 1)% 5].set_title('Level ' + str(col))
            if col == 3 or col == 8:
                axs[(col - 1) // 5, (col- 1) % 5].set_xlabel('Normalised and Scaled OFI')
            if col == 1 or col == 6:
                axs[(col - 1) // 5, (col- 1) % 5].set_ylabel('Density')
            xs = np.linspace(min(inputs[:, col-1:col]), max(inputs[:, col-1:col]), 100).flatten()
            p = torch.FloatTensor(norm.pdf(xs, torch.mean(val), torch.std(val)))
            axs[(col - 1) // 5, (col - 1)% 5].plot(xs, p, alpha=0.7, color='r')
        plt.savefig(instrument + '.png')
        plt.show()


def create_learning_curves_of_alpha_extraction():
    """Generates learning curves of alpha extraction during training."""
    fig, axs = plt.subplots(1, 5)

    fig.suptitle(f"Learning Curves of Alpha Extraction", fontsize=16)
    INSTRUMENTS = ["gold", "gbpusd", "eurusd", "uk100", "germany40"]
    LABELS = ["XAUUSD", "GBPUSD", "EURUSD", "FTSE100", "DE40"]
    for instrument in INSTRUMENTS:
        df = pd.read_csv('../data/' + instrument + '.csv')
        axs[INSTRUMENTS.index(instrument)].plot(list(df['Epoch'].values), list(df['Training Loss'].values), color='red', label='Training Loss' if instrument=="gold" else None)
        axs[INSTRUMENTS.index(instrument)].plot(list(df['Epoch'].values), list(df['Validation Loss'].values), color='blue', label='Validation Loss' if instrument=="gold" else None) 
        axs[INSTRUMENTS.index(instrument)].set_title(LABELS[INSTRUMENTS.index(instrument)])
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)
    axs[0].set_ylabel('MSE')
    axs[2].set_xlabel('Epoch')
    plt.show()


def create_reward_curves_of_reinforcement_learning_agents():
    """Generates reward curves of each reinforcement learning agents during training."""
    means = []
    stds = []
    vals = []
    
    AVERAGING = 5
    for agent in ["dqn", "ddqn", "qtable"]:
        fig, axs = plt.subplots(1, 5)
        fig.suptitle(f"Reward Curves of Q Learning", fontsize=20)
        INSTRUMENTS = ["gold", "gbpusd", "eurusd", "uk100", "germany40"]
        LABELS = ["XAUUSD", "GBPUSD", "EURUSD", "FTSE100", "DE40"]
        PIP_SIZE = [0.01, 0.0001, 0.0001, 0.1, 0.1]
        for instrument in INSTRUMENTS:
            df = pd.read_csv('../data/' + agent + '/' + instrument + '.csv')
            axs[INSTRUMENTS.index(instrument)].plot(list(range(0, 80, AVERAGING)), 
                                                    (conf.PRICE_TO_PNL[LABELS[INSTRUMENTS.index(instrument)]] * np.mean(np.array(df['Points Gained'].values).reshape(-1, AVERAGING), axis=1) / 1000.0).tolist()
                                                    , color='red', label='~PNL' if instrument=="gold" else None)

            axs[INSTRUMENTS.index(instrument)].scatter(list(range(0, 80)), 
                                                    (conf.PRICE_TO_PNL[LABELS[INSTRUMENTS.index(instrument)]] * np.array(df['Points Gained'].values) / 1000.0).tolist()
                                                    , color='red', marker='.', label='~PNL' if instrument=="gold" else None)
            axs[INSTRUMENTS.index(instrument)].axhline(y = 0, color = 'b', linestyle = '--')

            axs[INSTRUMENTS.index(instrument)].set_title(LABELS[INSTRUMENTS.index(instrument)])
        lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
        lines, labels = [sum(x, []) for x in zip(*lines_labels)]
        axs[0].set_ylabel('PNL (£1000s)')
        axs[2].set_xlabel('Episode')
        plt.show()


def capture_alpha_extractor_accuracy_metrics():
    """Calculates the RMSE, STD of data, out of sample R^2 metric of the alpha extractor."""
    scores = []
    rmses = []
    stds= []
    ross = []
    for instrument in ["XAUUSD", "GBPUSD", "EURUSD", "#UK100", "#Germany40"]:
        score = []
        rmse = []
        std = []
        ros = []
        pred=[]
        alpha_extractor = RegressionNetwork(layer_sizes=conf.SL_LAYER_SIZES)
        alpha_extractor.load_state_dict(torch.load('../models/alpha_extraction/' + instrument + '.pt'))
        alpha_extractor.eval()
        training_data, validation_data, testing_data = get_data_and_alphas(DATA_DIRECTORY + instrument + ".csv")[:3]
        testing_x, testing_y = convert_df_to_tensors(testing_data)
        for i in range(len(testing_x)):
            pred.append(alpha_extractor(testing_x[i]).detach().numpy().tolist())
        for col in range(6):
            p = np.array(pred)[:, col:col + 1].flatten().tolist()
            y = testing_y[:, col:col + 1].flatten().tolist()
            score.append(r2_score(y, p))
            benchmark = mean_squared_error(y, ([np.mean(np.array(y))] * len(y)))
            my_score = mean_squared_error(y,p)
            ros.append(1 - my_score/benchmark)
            rmse.append(sqrt(mean_squared_error(y,p)))
            std.append(np.std(np.array(y)))
        scores.append(score)
        rmses.append(rmse)
        ross.append(ros)
        stds.append(std)
    print(np.round(np.array(scores), 4).T)
    print("rmses:")
    print(np.round(np.array(rmses), 12).T)
    print("stds:")
    print(np.round(np.array(stds), 12).T)
    print("ros:")
    print(np.round(np.array(ross), 12).T)


# Used https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
# As a baseline
def create_qtable_heatmaps():
    """Creates heatmaps for Q table showing what alpha values at what horizons light up for reversing trades."""
    horizon_labels = ['Horizon 1', 'Horizon 2', 'Horizon 3', 'Horizon 4', 'Horizon 5', 'Horizon 6']
    bucket_labels = ["Very Neg-", "Slightly Neg-", "Neutral", "Slightly Pos+", "Very Pos+"]
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(f"Q Value Approximates for Q Learning", fontsize=20)
    INSTRUMENTS = ["XAUUSD", "GBPUSD", "EURUSD", "#UK100", "#Germany40"]
    q_values_short_to_long = np.zeros((conf.HORIZON, conf.SIZE_PER_BUCKET))
    q_values_long_to_short = np.zeros((conf.HORIZON, conf.SIZE_PER_BUCKET))
    q_vals_all = np.zeros((conf.SIZE_PER_BUCKET,) * conf.HORIZON + (2,))
    for short_to_long in [True, False]:
        for instrument in INSTRUMENTS:
            min_state_value_with_data = conf.MIN_STATE_VALUE_TRAINING[instrument]
            max_state_value_with_data = conf.MAX_STATE_VALUE_TRAINING[instrument]
            q_table = torch.Tensor(torch.load('../models/q_learning/'+ instrument + '.pt'))
            for i in range(len(horizon_labels)):
                dim = [0,1,2,3,4,5]
                dim.remove(i)
                q_vals_per_bucket = torch.sum(q_table[:,:,:,:,:,:,0 if short_to_long \
                                                      else 1,1 if short_to_long else 0], dim=dim) \
                                        .detach().numpy()
                if short_to_long:
                    q_values_short_to_long[i] += q_vals_per_bucket / np.linalg.norm(q_vals_per_bucket)
                else:
                    q_values_long_to_short[i] += q_vals_per_bucket / np.linalg.norm(q_vals_per_bucket)

    pcm = axs[1].imshow(q_values_long_to_short, cmap='Blues')
    axs[1].set_xticks(np.arange(len(bucket_labels)), labels=bucket_labels)
    axs[1].set_yticks(np.arange(len(horizon_labels)), labels=horizon_labels)
    plt.setp(axs[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
  
    for i in range(len(horizon_labels)):
        for j in range(len(bucket_labels)):
            text = axs[1].text(j, i, str(np.round(q_values_long_to_short[i, j], 3)),
                               ha="center", va="center", color="k")
    pcm = axs[0].imshow(q_values_short_to_long, cmap='Blues')
    axs[0].set_xticks(np.arange(len(bucket_labels)), labels=bucket_labels)
    axs[0].set_yticks(np.arange(len(horizon_labels)), labels=horizon_labels)
    plt.setp(axs[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
  
    for i in range(len(horizon_labels)):
        for j in range(len(bucket_labels)):
            text = axs[0].text(j, i, str(np.round(q_values_short_to_long[i, j], 3)),
                               ha="center", va="center", color="k")
    axs[1].set_title("Long to Short")
    axs[0].set_title("Short to Long")
    fig.colorbar(pcm, ax=axs)
    plt.show()


# Used https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
# As a baseline
def create_dqn_heatmaps():
    """Creates heatmaps for DQN showing what alpha values at what horizons light up for reversing trades."""
    horizon_labels = ['Horizon 1', 'Horizon 2', 'Horizon 3', 'Horizon 4', 'Horizon 5', 'Horizon 6']
    bucket_labels = ["Very Neg-", "Slightly Neg-", "Neutral", "Slightly Pos+", "Very Pos+"]
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(f"Q Value Approximates for Q Learning", fontsize=20)
    INSTRUMENTS = ["XAUUSD", "GBPUSD", "EURUSD", "#UK100", "#Germany40"]
    q_values_short_to_long = conf.Q_VALUES_DQN_SHORT_TO_LONG
    q_values_long_to_short = conf.Q_VALUES_DQN_LONG_TO_SHORT

    pcm = axs[1].imshow(q_values_long_to_short, cmap='Blues')
    axs[1].set_xticks(np.arange(len(bucket_labels)), labels=bucket_labels)
    axs[1].set_yticks(np.arange(len(horizon_labels)), labels=horizon_labels)
    plt.setp(axs[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
  
    for i in range(len(horizon_labels)):
        for j in range(len(bucket_labels)):
            text = axs[1].text(j, i, str(np.round(q_values_long_to_short[i, j], 3)),
                               ha="center", va="center", color="k")
    pcm = axs[0].imshow(q_values_short_to_long, cmap='Blues')
    axs[0].set_xticks(np.arange(len(bucket_labels)), labels=bucket_labels)
    axs[0].set_yticks(np.arange(len(horizon_labels)), labels=horizon_labels)
    plt.setp(axs[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
  
    for i in range(len(horizon_labels)):
        for j in range(len(bucket_labels)):
            text = axs[0].text(j, i, str(np.round(q_values_short_to_long[i, j], 3)),
                               ha="center", va="center", color="k")
    axs[1].set_title("Long to Short")
    axs[0].set_title("Short to Long")
    fig.colorbar(pcm, ax=axs)
    plt.show()


# Used https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
# As a baseline
def create_ddqn_heatmaps():
    """Creates heatmaps for DDQN showing what alpha values at what horizons light up for reversing trades."""
    horizon_labels = ['Horizon 1', 'Horizon 2', 'Horizon 3', 'Horizon 4', 'Horizon 5', 'Horizon 6']
    bucket_labels = ["Very Neg-", "Slightly Neg-", "Neutral", "Slightly Pos+", "Very Pos+"]
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(f"Q Value Approximates for Q Learning", fontsize=20)
    INSTRUMENTS = ["XAUUSD", "GBPUSD", "EURUSD", "#UK100", "#Germany40"]
    q_values_short_to_long = conf.Q_VALUES_DDQN_SHORT_TO_LONG
    q_values_long_to_short = conf.Q_VALUES_DDQN_LONG_TO_SHORT

    pcm = axs[1].imshow(q_values_long_to_short, cmap='Blues')
    axs[1].set_xticks(np.arange(len(bucket_labels)), labels=bucket_labels)
    axs[1].set_yticks(np.arange(len(horizon_labels)), labels=horizon_labels)
    plt.setp(axs[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
  
    for i in range(len(horizon_labels)):
        for j in range(len(bucket_labels)):
            text = axs[1].text(j, i, str(np.round(q_values_long_to_short[i, j], 3)),
                               ha="center", va="center", color="k")
    pcm = axs[0].imshow(q_values_short_to_long, cmap='Blues')
    axs[0].set_xticks(np.arange(len(bucket_labels)), labels=bucket_labels)
    axs[0].set_yticks(np.arange(len(horizon_labels)), labels=horizon_labels)
    plt.setp(axs[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
  
    for i in range(len(horizon_labels)):
        for j in range(len(bucket_labels)):
            text = axs[0].text(j, i, str(np.round(q_values_short_to_long[i, j], 3)),
                               ha="center", va="center", color="k")
    axs[1].set_title("Long to Short")
    axs[0].set_title("Short to Long")
    fig.colorbar(pcm, ax=axs)
    plt.show()
