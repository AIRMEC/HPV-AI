import os
import numpy as np
import pandas as pd
import random
import h5py
import pickle
import itertools
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler

from sklearn.utils.class_weight import compute_sample_weight

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)  

def define_data_sampling(train_split, val_split, method, workers):
    # Reproducibility of DataLoader.
    g = torch.Generator()
    g.manual_seed(0)

    # Set up training data sampler.
    if method == "random":
        print("random sampling setting")
        train_loader = DataLoader(
            dataset=train_split,
            batch_size=1,  # model expects one bag of features at the time.
            shuffle=True,
            collate_fn=collate,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        raise Exception(f"Sampling method '{method}' not implemented.")

    val_loader = DataLoader(
            dataset=val_split,
            batch_size=1,  # model expects one bag of features at the time.
            sampler=SequentialSampler(val_split),
            collate_fn=collate,
            num_workers=workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
    )

    return train_loader, val_loader


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def print_model(model):
    print(model)
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_trainable_params} parameters")
