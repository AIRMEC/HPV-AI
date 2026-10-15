import os
import numpy as np
import pandas as pd
import random
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, WeightedRandomSampler

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def collate_classificationtask(batch):
    #note if return things that don't need to be trained on, there is no need to call torch tensor, keep a numpy array.
    img = torch.cat([item[0] for item in batch], dim=0)
    label = torch.LongTensor([item[1] for item in batch])
    slide_id = [item[2] for item in batch]
    
    return [img, label, slide_id]

class FeatureBags_3Class_ClassificationDataset(Dataset):
    def __init__(self, df, data_dir, input_feature_size, class_to_idx):
        self.slide_df = df.copy().reset_index(drop=True)
        self.data_dir = data_dir
        self.input_feature_size = input_feature_size
        self.class_to_idx=class_to_idx
    
    def _get_feature_path(self, slide_id):
       
        return os.path.join(self.data_dir, f"{slide_id}_features.h5")

    def __getitem__(self, idx):
        slide_id = self.slide_df["slide_id"][idx]
        label = self.slide_df['HPV_p53_classification'][idx]
        cohort = self.slide_df["Cohort"][idx] if "Cohort" in self.slide_df else "NA"

        full_path = self._get_feature_path(slide_id)
        with h5py.File(full_path, "r") as hdf5_file:
            features = hdf5_file["features"][:]
            coords = hdf5_file["coords"][:]
        
        features = torch.from_numpy(features)[:,-self.input_feature_size:]

        return features, label, slide_id, cohort

    def __len__(self):
        return len(self.slide_df)   

def define_data_sampling(train_split, val_split, method, task, workers):
    # Reproducibility of DataLoader
    g = torch.Generator()
    g.manual_seed(0)

    # Set up training data sampler.
    if method == "random":
        print("random sampling setting")
        train_loader = DataLoader(
            dataset=train_split,
            batch_size=1,  # model expects one bag of features at the time.
            shuffle=True,
            collate_fn=collate_survivaltask if task=='prognostic' else collate_classificationtask,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    #only works with classification task
    elif method == "balanced":
        print("balanced sampling setting")
        train_labels = train_split.slide_df[f"HPV_p53_classification"]

        # Compute sample weights to alleviate class imbalance with weighted sampling.
        sample_weights = compute_sample_weight("balanced", train_labels)

        train_loader = DataLoader(
            dataset=train_split,
            batch_size=1,  # model expects one bag of features at the time.
            # Use the weighted sampler using the precomputed sample weights.
            # Note that replacement is true by default, so
            # some slides of rare classes will be sampled multiple times per epoch.
            shuffle=False,
            sampler=WeightedRandomSampler(sample_weights, len(sample_weights)),
            collate_fn=collate_classificationtask,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        raise Exception(f"Sampling method '{method}' not implemented.")

    # val_split would be an empty list if not validation is asked in training.
    if len(val_split) == 0:
        val_loader = val_split
    else:
        val_loader = DataLoader(
            dataset=val_split,
            batch_size=1,  # model expects one bag of features at the time.
            sampler=SequentialSampler(val_split),
            collate_fn=collate_survivaltask if task=='prognostic' else collate_classificationtask,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    return train_loader, val_loader 


def define_data_sampling_testing(val_split, task, workers):
    # Reproducibility of DataLoader
    g = torch.Generator()
    g.manual_seed(0)

    val_loader = DataLoader(
        dataset=val_split,
        batch_size=1,  # model expects one bag of features at the time.
        sampler=SequentialSampler(val_split),
        collate_fn=collate_classificationtask,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return val_loader

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def print_model(model):
    print(model)
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_trainable_params} parameters")

class MonitorBestModelEarlyStopping_3class_final:
    """Early stops the training if validation loss doesn't improve after a given patience and save best model """
    def __init__(self, patience=15, min_epochs=20, saving_checkpoint=True, foundation_model=None, hpset=None, output_dir="outputs"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            min_epochs (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.min_epochs = min_epochs
        self.counter = 0
        self.early_stop = False
        
        self.eval_loss_min = np.inf
        self.best_loss_score = None
        self.best_epoch_loss = None

        self.best_opt_metric_score = 0.0 #model will be optimized on this metric 
        self.best_epoch = None

        self.saving_checkpoint = saving_checkpoint

        self.foundation_model = foundation_model
        self.hpset = hpset
        self.output_dir = output_dir

    def __call__(self, epoch, eval_loss, eval_opt_metric, model, preds_intval, preds_train): 

        loss_score = -eval_loss
        opt_score = eval_opt_metric #AUC

        # Save model at epoch 0 and starts monitoring.
        if self.best_loss_score is None:
            self._update_loss_scores(loss_score, eval_loss, epoch)
            self._update_metrics_scores(opt_score, epoch)
            #self.save_checkpoint(model, log_dir, epoch)

        # Eval loss starts increasing. Recommend running early stopping on the loss.
        elif loss_score < self.best_loss_score:
            self.counter += 1
            print(f'Evaluation loss does not decrease : Starting Early stopping counter {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.min_epochs:
                self.early_stop = True
        # Eval loss keeps decreasing.
        else:
            print(f'Epoch {epoch} validation loss decreased ({self.eval_loss_min:.6f} --> {eval_loss:.6f})')
            self._update_loss_scores(loss_score, eval_loss, epoch)
            self._update_metrics_scores(opt_score, epoch)
            self.save_checkpoint_predictions(model, preds_intval, preds_train)
            self.counter = 0
        
        # We may have a tiny lag between min loss and the best C-index. With the patience and early stop, it is fine but better to save based on highest C-index too. 
        if opt_score > self.best_opt_metric_score:
            self.save_checkpoint_predictions(model, preds_intval, preds_train)

    def save_checkpoint_predictions(self, model, preds_intval, preds_train):
        print(f"Saving checkpoint and predictions")
        #Checkpoints
        file_dir_check = os.path.join(self.output_dir, 'checkpoints')
        filepath_check = os.path.join(file_dir_check, f"HERMES_checkpoint.pt")
        os.makedirs(file_dir_check, exist_ok=True)
        torch.save(model.state_dict(), filepath_check)

        #Predictions validation
        file_dir_pred_intval = os.path.join(self.output_dir, 'predictions')
        filepath_pred_intval = os.path.join(file_dir_pred_intval, "HERMES_INTVAL_predictions.csv")
        os.makedirs(file_dir_pred_intval, exist_ok=True)
        preds_intval.to_csv(filepath_pred_intval, index=False)       

        #Predictions training
        file_dir_pred_train = os.path.join(self.output_dir, 'predictions')
        filepath_pred_train = os.path.join(file_dir_pred_train, "HERMES_TRAIN_predictions.csv")
        preds_train.to_csv(filepath_pred_train, index=False)              

    def _update_loss_scores(self, loss_score, eval_loss, epoch):
        self.eval_loss_min = eval_loss
        self.best_loss_score = loss_score
        self.best_epoch_loss = epoch
        print(f'Updating loss at epoch {self.best_epoch_loss} -> {self.eval_loss_min:.6f}')
    
    def _update_metrics_scores(self, opt_score, epoch):
        # Even though loss would decrease, metric does not necessarily increase.
        self.best_opt_metric_score = opt_score
        self.best_epoch = epoch
        print(f'Updating Opt metric at epoch {self.best_epoch} -> {self.best_opt_metric_score:.6f}')