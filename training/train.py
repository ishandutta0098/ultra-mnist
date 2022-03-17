import gc 
import time 
import copy 
import torch
import wandb
import training.engine as engine 
from collections import defaultdict
from tqdm.auto import tqdm

import data.dataloader as dataloader
import utilities.utils as utils

from torch.cuda import amp 

def run_training(args, cfg, model, optimizer, scheduler, weights, device, num_epochs, df_train, df_valid, oof_csv, base_path):
    """
    Function to run the training and validation on a fold of data

    Args:
        args: Argparse Arguments 
        cfg (dict): Configuration file
        model (PyTorch Model): Model Class
        optimizer: Optimizer for the network
        scheduler: Learning Rate Scheduler
        weights (torch tensor): Class Weight values
        device (torch.device): GPU or CPU
        num_epochs (int): Number of Epochs
        df_train (pandas dataframe): Training DataFrame
        df_valid (pandas dataframe): Validation DataFrame
        oof_csv (str): Path to save the validation predictions
        base_path (str): Base path for the system
        
    """
    
    start = time.time()
    history = defaultdict(list)

    best_acc = 0

    # Get dataloaders
    train_loader, valid_loader, valid_ids = dataloader.prepare_loaders(
                                                            cfg, 
                                                            df_train, 
                                                            df_valid
                                                        )

    # Initialize weights and biases
    run = wandb.init(
        project=cfg['MODEL']['PROJECT_NAME'], 
        config = cfg,
        group = cfg['MODEL']['GROUP_NAME'] 
        )

    # Set wandb run name
    wandb.run.name = cfg['MODEL']['RUN_NAME']
    wandb.watch(model)

    wandb.log(
        {
            'fold': args.fold,
        }
    )

    # Use amp scaler if model is running 
    # on a cuda enabled device
    if device == torch.device('cuda'):

        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))

        # Scaler for Automatic Mixed Precision
        scaler = amp.GradScaler()

    else:

        print('\nTraining on CPU')
        scaler = None
    
    # Run training and validation for given epochs
    for epoch in range(1, num_epochs + 1): 
        print("\n---------------------------------------------------------")
        print(f"########## Epoch: {epoch}/{cfg['TRAIN']['EPOCHS']} ##########")
        print("---------------------------------------------------------")
        print()

        gc.collect()

        # Training
        train_epoch_loss, train_acc = engine.train_one_epoch(
                                        cfg,
                                        model, 
                                        optimizer, 
                                        scheduler, 
                                        dataloader=train_loader, 
                                        device=device, 
                                        epoch=epoch,
                                        weights=weights,
                                        scaler=scaler
                                        )
        
        # Validation
        val_epoch_loss, val_acc, targets, outputs = engine.valid_one_epoch(
                                        cfg,
                                        model, 
                                        optimizer, 
                                        valid_loader, 
                                        device=device, 
                                        epoch=epoch,
                                        weights=weights
                                        )


        # Save model based on validation accuracy
        if val_acc > best_acc:
            print(f'>> Validation Accuracy Improved - Val Acc: Old: {best_acc} | New: {val_acc}')
            best_model = copy.deepcopy(model)
            best_acc = val_acc

            best_model_path = base_path + cfg['MODEL']['MODEL_PATH'] + "/" + cfg['MODEL']['RUN_NAME'] + "_" + args.fold + "_acc_" + str(best_acc) + ".bin"

            pred = outputs
        
        # Log metrics
        wandb.log(
            {
            'epoch': epoch,
            'train_epoch_loss': train_epoch_loss,
            'train_acc': train_acc,
            'val_epoch_loss': val_epoch_loss,
            'val_acc': val_acc,
            'best_acc': best_acc
            }
        )

        # Store the training history
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)

        print(f"> Epochs: {epoch}/{num_epochs} - Train Loss: {train_epoch_loss} - Train Acc: {train_acc} - Val Loss: {val_epoch_loss} - Val Acc: {val_acc}")
        print()

    print(f'>> Saving Best Model with Val Acc: {best_acc}')
    print()

    torch.save(
            best_model.state_dict(), 
            best_model_path
        )
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))

    wandb.log(
        {
        'training_time(mins)': time_elapsed/60
        }
    )

    utils.save_oof(
        id = valid_ids, 
        target = targets, 
        pred = pred, 
        fold = args.fold, 
        oof_file_path = oof_csv
        )

    wandb.finish()