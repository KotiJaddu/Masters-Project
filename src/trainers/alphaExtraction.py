""""""""""""""""""""
#  Trains a linear regression model (supervised learning component) to extract alphas at 
#  multiple horizons from the order flow imbalance features, then tests on unseen data.
""""""""""""""""""""
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import config as conf
from utils import get_data_and_alphas, RegressionNetwork

DEVICE = None

# Use GPU if available
if conf.USE_GPU and torch.cuda.is_available():
    DEVICE = torch.device('cuda:0')
else:
    DEVICE = torch.device('cpu')


def fit(training_dl, validation_dl, model, early_stopping, loss_fn, opt, verbose):
    """
    Runs the training and testing for the model.
    
    Args:
            training_dl: a DataLoader containing the training dataset
            validation_dl: a DataLoader containing the validation dataset
            model: the regression model to train
            early_stopping: early stopping patience hyperparameter
            loss_fn: the loss function (MSE)
            opt: the optimizer used to update the parameter values (ADAM)
            verbose: prints updates to the command line
    Returns:
            average validation loss (for hyperparameter tuning)
    """
    model = model.to(device=DEVICE)
    k = 0
    best_val_score = 30000 # set high value
    epoch = 0
    # Train until epoch limit is exceeded or due to early stopping
    while k < early_stopping and epoch < conf.SL_MAXIMUM_NUMBER_OF_EPOCHS:
        epoch += 1
        # Training
        total_steps = 0
        total_loss = 0.0
        for xb,yb in training_dl:
            model.train()
            total_steps += len(xb)
            xb = xb.to(device=DEVICE, dtype=conf.DTYPE)
            yb = yb.to(device=DEVICE, dtype=conf.DTYPE)
            pred = model(xb)
            loss = loss_fn(pred, yb, reduction="sum")
            # Perform back propagation step
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            # Calculate metrics for printing
            average_loss = total_loss / total_steps
            if total_steps % (len(xb) * 1000) == 0 and verbose:
                average_loss = total_loss / total_steps
                print(f'Epoch {epoch}\t\t Training loss: {average_loss:.12f}\t ' + \
                      f'{total_steps//len(xb)}/{len(training_dl)}', end='\r')

        # Validation
        model.eval()
        total_val_steps = 0
        total_val_loss = 0.0
        for xb,yb in validation_dl:
            total_val_steps += len(xb)
            xb = xb.to(device=DEVICE, dtype=conf.DTYPE)
            yb = yb.to(device=DEVICE, dtype=conf.DTYPE)
            pred = model(xb)
            loss = loss_fn(pred, yb, reduction="sum")
            total_val_loss += loss.item()
        average_val_loss = total_val_loss / total_val_steps
        # Early stopping
        if best_val_score <= average_val_loss:
            k += 1
        else:
            k = 0
            best_val_score = average_val_loss
        if verbose:
            print(f'Epoch {epoch}\t\t Training loss: {average_loss:.12f}' + \
                  f'\t Validation loss: {average_val_loss:.12f}\n')
    return average_val_loss


def convert_df_to_tensors(df):
    """
    Splits a Pandas dataframe into x, y tensors (inputs and labels).
    
    Args:
            df: a Pandas dataframe containing both inputs and labels
    Returns:
            inputs and labels as tensors
    """
    # Get order flow imbalance data
    x = torch.FloatTensor([[0] * len(i) if max(abs(np.array(i))) == 0 else
                           (np.array(i) / max(abs(np.array(i)))).tolist() for i in
                           [list(map(float, y)) for y in
                            [x[0][1:-1].split(',') for x in df[['OFI']].values]]])
    # Get alpha data
    y = torch.Tensor(df.drop(['Time', 'OFI', 'Mid Price', 'Date'], axis=1).values)
    return (x, y)


def run(data_path, parameters={"LAYER_SIZES": conf.SL_LAYER_SIZES,
                               "LEARNING_RATE": conf.SL_LEARNING_RATE,
                               "EARLY_STOPPING": conf.SL_EARLY_STOPPING,
                               "BATCH_SIZE": conf.SL_BATCH_SIZE},
        save_model=True, verbose=True):
    """
    Runs the training and testing for the model.
    
    Args:
            data_path: path to the OFI data
            parameters: hyperparameters for the model
            save_model: saves the model to disk
            verbose: prints updates to the command line
    Returns:
            average validation loss (for hyperparameter tuning)
    """
    instrument = data_path.split('/')[-1].split('.')[0]
    # Load data
    if verbose:
        print("Loading data; using " + str(DEVICE))
    training_data, validation_data, testing_data = get_data_and_alphas(data_path)[:3]
    if verbose:
        print("Transforming data")
    instrument = data_path.split('/')[-1].split('.')[0]
    training_x, training_y = convert_df_to_tensors(training_data)
    validation_x, validation_y = convert_df_to_tensors(validation_data)
    testing_x, testing_y = convert_df_to_tensors(testing_data)
    
    training_ds = TensorDataset(training_x, training_y)
    validation_ds = TensorDataset(validation_x, validation_y)
    testing_ds = TensorDataset(testing_x, testing_y)
    
    training_dl = DataLoader(training_ds, parameters["BATCH_SIZE"], shuffle=True)
    validation_dl = DataLoader(validation_ds, parameters["BATCH_SIZE"], shuffle=False)
    testing_dl = DataLoader(testing_ds, parameters["BATCH_SIZE"], shuffle=False)
    
    # Create model
    model = RegressionNetwork(parameters["LAYER_SIZES"])
    opt = torch.optim.Adam(model.parameters(), lr=parameters["LEARNING_RATE"][instrument],
                           weight_decay=conf.L2_REGULARISATION)
    loss_fn = F.mse_loss

    # Train model
    if verbose:
        print("Fitting model")
    average_val_loss = fit(training_dl, validation_dl, model, parameters["EARLY_STOPPING"],
                           loss_fn, opt, verbose)
    # Save model
    if save_model:
        torch.save(model.state_dict(), "../models/alpha_extraction/" + instrument + ".pt")
    
    # Test the model
    total_testing_steps = 0
    total_testing_loss = 0.0
    for xb,yb in testing_dl:
        total_testing_steps += len(xb)
        xb = xb.to(device=DEVICE, dtype=conf.DTYPE)
        yb = yb.to(device=DEVICE, dtype=conf.DTYPE)
        pred = model(xb)
        loss = loss_fn(pred, yb, reduction="sum")
        total_testing_loss += loss.item()
    average_testing_loss = total_testing_loss / total_testing_steps
    if verbose:
        print(f'Testing loss: {average_testing_loss:.12f}\n')
    return average_val_loss # Return validation loss for hyperparameter tuning
        

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m trainers.alphaExtraction path_to_ofi_data")
        sys.exit(0)
    run(sys.argv[1])
