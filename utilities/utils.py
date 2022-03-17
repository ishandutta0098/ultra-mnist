# Project Utility File
import os
import yaml
import random

import pandas as pd 
import numpy as np 

from sklearn.model_selection import StratifiedKFold
import torch

def load_config(config_file):
    """
    Function to load a yaml configuration file.
    
    Args:
        config_file (str): Name of the configuration file
        
    Returns: 
        config (dict): yaml config dict
    """
    
    config_path = os.path.join('configuration', config_file)
    
    with open (config_path) as file:
        config = yaml.safe_load(file)
        
    return config

def set_seed(SEED=42):
    """
    Set the seed for reproducibility.
    
    Args:
        SEED (int, optional): Seed value to be set. Defaults to 42.
    """
    
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(SEED)
    
set_seed()

def get_skfold(df):
    """
    Function to split the dataset into stratified kfolds
    
    Args: 
        df (pandas dataframe): Dataframe to generate folds
    
    Returns:
        df (pandas dataframe): DataFrame with folds
    """
    
    skf = StratifiedKFold(
                    n_splits = 5,
                    shuffle = True,
                    random_state = 42
                    )
    
    for fold, (_, val_) in enumerate(skf.split(X = df, y=df.target)):
        df.loc[val_, "fold"] = fold
        
    return df

def save_oof(id, target, pred, fold, oof_file_path):
    """
    Function to save out of fold predictions.
    
    Args:
        id (list): List of image_ids
        target (list): List of Targets
        pred (list): List of model predictions
        fold (int): Data fold
        oof_file_path (str): Path for the OOF File
    """
    
    df = pd.DataFrame()
    df['image_id'] = id 
    df['target'] = target
    df['pred'] = pred 
    df['fold'] = fold
    
    if fold == 0:
        df.to_csv(
            oof_file_path, 
            index = False
        )
        
    else:
        df_oof = pd.read_csv(oof_file_path)
        df_oof = df_oof.append(df, ignore_index = True)
        df_oof.to_csv(oof_file_path, index = False)
        
def save_valid_preds(id, target, pred, pred_file_path):
    """
    Function to save the model predictions.

    Args:
        id (list): List of image ids
        target (list): List of targets
        pred (list): List of model predictions
        pred_file_path (str): Path for the prediction file
    """

    df = pd.DataFrame()
    df['image_id'] = id
    df['target'] = target 
    df['pred'] = pred

def get_oof_score(oof_file_path):
    
    df = pd.read_csv(oof_file_path)
    
    pred = list(df.pred)
    target = list(df.target)
    
    accuracy = (np.sum(np.array(target) == np.array(pred))/len(target)) * 100
    
    print(f"Out of fold Accuracy: {accuracy}")

def get_image_path(df, data):
    """
    Function to obtain the image paths.
    
    Args: 
        df (pandas dataframe): Dataframe with image ids
        data (str): Train or Test Data
    
    Returns:
        df (pandas dataframe): DataFrame with Image Paths
    """
    
    df['image_path'] = ["input/" + data + "/" + x + ".jpeg" for x in df['id']]
    
    return df

def get_train_val_data(train_path, fold):

    df = pd.read_csv(train_path)

    df_train = df[df.fold != fold].reset_index(drop=True)
    df_valid = df[df.fold == fold].reset_index(drop=True)

    return df_train, df_valid