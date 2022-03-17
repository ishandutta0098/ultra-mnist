import pandas as pd 
import cv2 
import gc 
import torch 
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
import numpy as np
import argparse

import data.dataloader as dataloader
import data.augmentations as augmentations 

import utilities.utils as utils
import utilities.model_utils as model_utils

def run_inference(args):

    print("\n---------------------------------------------------------")
    print(f"########## Initialize Model Testing ##########")
    print("---------------------------------------------------------")
    print()

    # Load Configuration File
    cfg = utils.load_config(args.config)

    # Get paths 
    base_path = utils.get_base_path(args, cfg) 
    test_path = base_path + cfg['DATA']['TEST_CSV']
    pred_path = base_path + cfg['PREDICT']['TEST_CSV']

    df = pd.read_csv(test_path)

    # Get dataloader
    test_loader, test_ids = dataloader.prepare_test_loader(cfg, df)

    # Setup model
    device = torch.device(args.device)
    model = model_utils.setup_pretrained_model(cfg, base_path, device)
    model.to(device)

    print("\n---------------------------------------------------------")
    print(f"########## Model and Data Prepared ##########")
    print("---------------------------------------------------------")
    print()

    def test_fn(model, dataloader, device):
        """
        Function to test model accuracy

        Args:
            model: Model definition
            dataloader: PyTorch DataLoader
            device: GPU or CPU device for training

        Returns:
            final_targets (list): List of Targets
            final_outputs (list): List of Model Predictions
        """

        print("\n---------------------------------------------------------")
        print(f"########## Starting Evaluation ##########")
        print("---------------------------------------------------------")
        print()

        model.eval()

        final_outputs = []

        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for step, data in bar:   

            print(step)

            # Get data
            images = data['image'].to(device, dtype=torch.float)

            # Generate predictions
            outputs = model(images)

            # Find class with highest output
            _ , pred = torch.max(outputs.data, 1)

            # Move targets and outputs to cpu 
            outputs = (pred.detach().cpu().numpy()).tolist()

            final_outputs.extend(outputs)

        gc.collect()
        
        return final_outputs

    outputs = test_fn(model, test_loader, device)

    # Save test predictions
    utils.save_test_preds(
        test_ids, 
        outputs, 
        pred_file_path = cfg['PREDICT']['TEST_CSV']],
    )

    print("\n---------------------------------------------------------")
    print(f"########## Model Testing Complete ##########")
    print("---------------------------------------------------------")
    print()

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--device', type=str, required=True, help='cuda or cpu')
    parser.add_argument('--environ', type=str, required=True, help='colab or local')
    
    args = parser.parse_args()

    run_inference(args)