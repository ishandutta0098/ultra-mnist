import os
from re import M
import pandas as pd
import torch
import argparse
import wandb 
from tqdm.auto import tqdm

import utilities.utils as utils
import training.train as train 
import utilities.model_utils as model_utils

if __name__ == "__main__":

    # Login to weights and biases
    wandb.login()

    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, required=True, help='cuda or cpu')
    parser.add_argument('--environ', type=str, required=True, help='colab or local')
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--check', type=bool)

    args = parser.parse_args()

    # Load Configuration File
    cfg = utils.load_config(args.config)
    
    if args.environ == 'colab':
        # mount drive at drive.mount('/drive)
        BASE_PATH = cfg['DATA']['COLAB_BASE_PATH']

    elif args.environ == 'local':
        BASE_PATH = cfg['DATA']['BASE_PATH']

    # Get paths
    TRAIN_PATH = BASE_PATH + cfg['DATA']['TRAIN_CSV']
    OOF_PATH = BASE_PATH + cfg['DATA']['OOF_CSV']

    # Device
    DEVICE = torch.device(args.device)

    # Make model
    model, optimizer, scheduler = model_utils.make_model(cfg)
    model.to(DEVICE)

    # Load data
    df_train, df_valid = utils.get_train_val_data(
                                            TRAIN_PATH, 
                                            args.fold
                                            )

    # Get Class weights for calculating Loss
    if cfg['TRAIN']['WEIGHTS'] == True:
        weights = utils.get_class_weights(df_train)

    else: 
        weights = None

    if args.check == True:
        df_train = df_train.head(1000)

    # Train and validate model

    print("\n---------------------------------------------------------")
    print(f"########## Initialize Training ##########")
    print("---------------------------------------------------------")
    print()

    train.run_training(
        args,
        cfg,
        model, 
        optimizer, 
        scheduler,
        weights = weights,
        device=DEVICE,
        num_epochs=cfg['TRAIN']['EPOCHS'],
        df_train = df_train,
        df_valid = df_valid,
        oof_csv=OOF_PATH,
        base_path = BASE_PATH    
    )

    print("\n---------------------------------------------------------")
    print(f"########## Training Completed ##########")
    print("---------------------------------------------------------")
    print()