# Project Utility File

import os
import yaml
import random

import pandas as pd 
import numpy as np 

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
    
    with open (config_path) ass file:
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
        
def get_oof_score(oof_file_path):
    
    df = pd.read_csv(oof_file_path)
    
    pred = list(df.pred)
    target = list(df.target)
    
    accuracy = (np.sum(np.array(target) == np.array(pred))/len(target)) * 100
    
    print(f"Out of fold Accuracy: {accuracy}")
        