import os
import random
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics import roc_auc_score
import numpy as np
import torch

from HERA import HERA
from utils_HERA import *

def set_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def evaluate_model(epoch, model, device, loader, writer, loss_fn):
    model.eval()
    eval_loss = 0.

    probs = np.zeros(len(loader))
    labels = np.zeros(len(loader), dtype=int)
    slide_ids = []
    cohorts = []

    with torch.no_grad():
        for batch_idx, (data, label, slide_id, cohort) in enumerate(loader):
            data, label = data.to(device), label.to(device).float()
            slide_id = slide_id[0]

            logit, Y_prob, _, _ = model(data)
            logit = logit.squeeze(dim=1)  
            Y_prob = Y_prob.squeeze(dim=1)

            loss = loss_fn(logit, label)
            eval_loss += loss.item()

            probs[batch_idx] = Y_prob.cpu().item()
            labels[batch_idx] = label.cpu().item()
            slide_ids.append(slide_id)
            cohorts.append(cohort[0])

    eval_loss /= len(loader)

    auc = roc_auc_score(labels, probs)
    print(f'Eval epoch: {epoch}, loss: {eval_loss}, AUC: {auc:.4f}')

    # Compute AUC per cohort
    unique_cohorts = set(cohorts)
    print("Eval AUC per cohort:")
    for c in unique_cohorts:
        idxs = [i for i, coh in enumerate(cohorts) if coh == c]
        c_labels = [labels[i] for i in idxs]
        c_probs = [probs[i] for i in idxs]
        n_samples = len(c_labels)
        if len(set(c_labels)) > 1:  # Only compute AUC if both classes are present
            c_auc = roc_auc_score(c_labels, c_probs)
            print(f"  {c}: AUC={c_auc:.4f}, n_samples={n_samples}")
        else:
            print(f"  {c}: only one class present, n_samples={n_samples}")   

    writer.add_scalar("Loss/eval", eval_loss, epoch)
    writer.add_scalar("auc/eval", auc, epoch)

    return eval_loss, auc, probs, labels, slide_ids

def train_one_epoch(epoch, model, device, train_loader, optimizer, writer, loss_fn):
    
    model.train()
    epoch_start_time = time.time()
    train_loss = 0.

    probs = np.zeros(len(train_loader))
    labels = np.zeros(len(train_loader))
    slide_ids = []
    cohorts = []

    batch_start_time = time.time()
    for batch_idx, (data, label, slide_id, cohort) in enumerate(train_loader):
        data_load_duration = time.time() - batch_start_time

        data, label = data.to(device), label.to(device).float()

        logit, Y_prob, _, _ = model(data)
        logit = logit.squeeze(dim=1) 
        Y_prob = Y_prob.squeeze(dim=1)  

        loss = loss_fn(logit, label)
        train_loss += loss.item()

        probs[batch_idx] = Y_prob.detach().cpu().item()
        labels[batch_idx] = label.cpu().item()
        slide_ids.append(slide_id)
        cohorts.append(cohort[0])

        # backward pass
        loss.backward()

        # step
        optimizer.step()
        optimizer.zero_grad()

        batch_duration = time.time() - batch_start_time
        batch_start_time = time.time()

        writer.add_scalar("duration/data_load", data_load_duration, epoch)
        writer.add_scalar("duration/batch", batch_duration, epoch)

    epoch_duration = time.time() - epoch_start_time
    print(f"Finished training on epoch {epoch} in {epoch_duration:.2f}s")

    train_loss /= len(train_loader)

    # Compute Overall AUC
    auc = roc_auc_score(labels, probs)
    print(f'Train epoch: {epoch}, loss: {train_loss}, AUC: {auc:.4f}')

    # Compute AUC per cohort
    unique_cohorts = set(cohorts)
    print("Train AUC per cohort:")
    for c in unique_cohorts:
        idxs = [i for i, coh in enumerate(cohorts) if coh == c]
        c_labels = [labels[i] for i in idxs]
        c_probs = [probs[i] for i in idxs]
        n_samples = len(c_labels)
        if len(set(c_labels)) > 1:  # Only compute AUC if both classes are present
            c_auc = roc_auc_score(c_labels, c_probs)
            print(f"  {c}: AUC={c_auc:.4f}, n_samples={n_samples}")
        else:
            print(f"  {c}: only one class present, n_samples={n_samples}")   

    writer.add_scalar("duration/epoch", epoch_duration, epoch)
    writer.add_scalar("LR", get_lr(optimizer), epoch)
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("auc/train", auc, epoch)

    return probs, labels, slide_ids

def run_train_eval_loop(train_loader, val_loader, loss_fn, hparams, run_id, foundation_model, hpset, output_dir):
    print(f"Predicting HPV status")
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(output_dir, "runs", run_id))
    device = torch.device("cuda")

    model = HERA(
        input_feature_size=hparams["input_feature_size"],
        precompression_layer=hparams["precompression_layer"],
        feature_size_comp=hparams["feature_size_comp"],
        feature_size_attn=hparams["feature_size_attn"],
        postcompression_layer=hparams["postcompression_layer"],
        feature_size_comp_post=hparams["feature_size_comp_post"],
        dropout=True,
        p_dropout_fc=hparams["p_dropout_fc"],
        p_dropout_atn=hparams["p_dropout_atn"],
    ).to(device)
    print_model(model)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=hparams["initial_lr"],
        weight_decay=hparams["weight_decay"],  
        betas=(0.9, 0.999),                     # same defaults as Adam
        eps=1e-8
    )

    # Using a multi-step LR decay routine.
    if hparams["milestones"]=='cos':
        # Usina cosine decay routine 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=hparams["max_epochs"], 
            eta_min=0.0, 
            last_epoch=-1, 
            verbose=True)
    else:
        milestones = [int(x) for x in hparams["milestones"].split(",")]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=hparams["gamma_lr"]
        )

    monitor_tracker = MonitorBestModelEarlyStopping_Binary(
        patience=hparams["earlystop_patience"],
        min_epochs=hparams["earlystop_min_epochs"],
        saving_checkpoint=True,
        foundation_model=foundation_model,
        hpset=hpset,
        output_dir=output_dir
    )

    for epoch in range(hparams["max_epochs"]):

        probs_train, labels_train, slide_ids_train = train_one_epoch(epoch, model, device, train_loader, optimizer, writer, loss_fn)

        preds_train = pd.DataFrame(probs_train, columns=[f'prob_HPV'])
        preds_train['label'] = labels_train
        preds_train['slide_id'] = slide_ids_train

        # Evaluation. 
        eval_loss, eval_auc, probs, labels, slide_ids = evaluate_model(epoch, model, device, val_loader, writer, loss_fn) 

        preds_intval = pd.DataFrame(probs, columns=[f'prob_HPV'])
        preds_intval['label'] = labels
        preds_intval['slide_id'] = slide_ids

        monitor_tracker(epoch, eval_loss, eval_auc, model, preds_intval, preds_train)

        # Update LR decay.
        scheduler.step()

        if monitor_tracker.early_stop:
            print(f"Early stop criterion reached. Broke off training loop after epoch {epoch}.")
            break
    
    # Log the hyperparameters of the experiments
    runs_history = {
        "run_id" : run_id,
        "best_epoch_loss": monitor_tracker.best_epoch_loss,
        "best_evalLoss" : monitor_tracker.eval_loss_min,
        "best_epoch_AUC" : monitor_tracker.best_epoch,
        "best_AUC_score" : monitor_tracker.best_opt_metric_score,
        **hparams,
    }

    with open(os.path.join(output_dir, 'run_history_HERA.txt'), 'a') as filehandle:
        for _, value in runs_history.items():
            filehandle.write('%s;' % value)
        filehandle.write('\n')

    writer.close()


def main(args):

    # Set random seed for some degree of reproducibility. See PyTorch docs on this topic for caveats.
    # https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
    set_seed()

    if not torch.cuda.is_available():
        raise Exception(
            "No CUDA device available. Training without one is not feasible."
        )

    df_train = pd.read_csv(args.manifest_training)
    print(f'Read {args.manifest_training} training dataset containing {len(df_train)} samples')

    df_val = pd.read_csv(args.manifest_intval)
    print(f'Read {args.manifest_intval} internal validation dataset containing {len(df_val)} samples')

    try:
        training_set = df_train
        validation_set = df_val
    except:
        raise Exception(
            f"Column fold- does not exist in {args.manifest}"
        )

    train_split = FeatureBags_Binary_ClassificationDataset(df=training_set,
                                    data_dir=args.data_dir,
                                    input_feature_size=args.input_feature_size)

    val_split = FeatureBags_Binary_ClassificationDataset(df=validation_set,
                                    data_dir=args.data_dir, 
                                    input_feature_size=args.input_feature_size)
    

    try:
        git_sha = (
            subprocess.check_output(["git", "describe", "--always"], stderr=subprocess.DEVNULL).strip().decode("utf-8")
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_sha = "nogit"
    train_run_id = f"{git_sha}-{time.strftime('%Y%m%d-%H%M%S')}"

    print(f"=> Git SHA {train_run_id}")
    print(f"=> Training on {len(train_split)} samples")
    print(f"=> Validating on {len(val_split)} samples")

    base_hparams = dict( 
        input_feature_size=args.input_feature_size,
        features_extraction=os.path.dirname(args.data_dir),

        # Settings that be changed in the loop:
        # Training.
        sampling_method="random",
        class_weighting=False,
        max_epochs=30,
        earlystop_patience=30, 
        earlystop_min_epochs=30, 

        # Optimizer.
        initial_lr=0.00003,
        milestones="5, 15, 25",
        gamma_lr=0.1,
        weight_decay=0.00001,

        # Model architecture parameters. See model class for details.
        precompression_layer=False,
        feature_size_comp=512,
        feature_size_attn=256,
        postcompression_layer=True,
        feature_size_comp_post=128,
        p_dropout_fc=0.25,
        p_dropout_atn=0.25,
    )

    hparam_sets = [

        {   ### hp 0 ###
            **base_hparams,
        },
        {   ### hp 1 ###
            **base_hparams,
            'p_dropout_fc': 0.5,
            'p_dropout_atn': 0.5
        },
        {   ### hp 2 ###
            **base_hparams,
            'feature_size_attn': 128
        },
        {   ### hp 3 ###
            **base_hparams,
            'feature_size_comp_post': 64
        },
        {   ### hp 4 ###
            **base_hparams,
            'class_weighting': True
        },
        {   ### hp 5 ###
            **base_hparams,
            'sampling_method': "balanced"
        },
        {   ### hp 6 ###
            **base_hparams,
            'feature_size_attn': 128,
            'sampling_method': "balanced",
            'class_weighting': True
        },
        {   ### hp 7 ###
            **base_hparams,
            'feature_size_attn': 128,
            'milestones':"5, 10, 15",
            'gamma_lr':0.4,
            'weight_decay':0.0001,
        },
        {   ### hp 8 ###
            **base_hparams,
            'feature_size_attn': 128,
            'initial_lr':0.00002,
            'milestones':"5, 10, 15",
            'gamma_lr':0.5,
            'weight_decay':0.0005,
        },
        {   ### hp 9 ###
            **base_hparams,
            'feature_size_attn': 128,
            'initial_lr':0.00004,
            'milestones':"5, 12, 20",
            'gamma_lr':0.35,
            'weight_decay':0.0002,
        },
        {   ### hp 10 ###
            **base_hparams,
            'feature_size_attn': 128,
            'initial_lr':0.00004,
            'milestones':"5, 12, 20",
            'gamma_lr':0.35,
            'weight_decay':0.0002,
            'p_dropout_fc':0.35,
            'p_dropout_atn':0.35,
        },
        {   ### hp 11 ### --> new optimizer AdamW requires slitghly lower weight decay!
            **base_hparams,
            'feature_size_attn': 128,
            'initial_lr':0.00004,
            'milestones':"5, 12, 20",
            'gamma_lr':0.35,
            'weight_decay':0.00005,
            'max_epochs':100,
        },
        {   ### hp 12 ### --> new optimizer AdamW requires slitghly lower weight decay!
            **base_hparams,
            'feature_size_attn': 128,
            'initial_lr':0.00004,
            'milestones':"10, 20",
            'gamma_lr':0.35,
            'weight_decay':0.0005,
        },
    ]

    hps = hparam_sets[args.hp]
    print(hps)

    run_id = f"{args.foundationtype}_final_{train_run_id}_hp{args.hp}"
    print(f"Running train-eval loop {args.hp} for {run_id}")
    print(hps)

    train_loader, val_loader = define_data_sampling(
            train_split,
            val_split,
            method=hps["sampling_method"],
            task='classification',
            workers=args.workers
    )

    device = torch.device("cuda")

    if hps["class_weighting"] == True:
        all_labels = [train_split[i][1] for i in range(len(train_split))]
        all_labels = torch.tensor(all_labels)

        # Compute positive and negative counts
        num_pos = (all_labels == 1).sum()
        num_neg = (all_labels == 0).sum()

        # Compute pos_weight for BCEWithLogitsLoss
        pos_weight = num_neg / num_pos
        print("The pos_weight in BCEWithLogitsLoss is: ", pos_weight)

        lossfunction = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    else:
        lossfunction = nn.BCEWithLogitsLoss()    

    run_train_eval_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn = lossfunction,
            hparams=hps,
            run_id=run_id,
            foundation_model = args.foundationtype, 
            hpset = args.hp,
            output_dir = args.output_dir,
    )
    print("Finished training.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--manifest_training",
        type=str,
        help="CSV file listing all slides for training final model, their labels, and which split (train/test/val) they belong to.",
    )
    parser.add_argument(
        "--manifest_intval",
        type=str,
        help="CSV file listing all slides in internal validation set, their labels, and which split (train/test/val) they belong to.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory where all *_features.h5 files are stored",
    )
    parser.add_argument(
        "--input_feature_size",
        help="The size of the input features from the feature bags. Recommend going by blocks from these output size [96, 96, 192, 192, 384, 384, 384, 384, 768, 768]",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/HERA",
        help="Where checkpoints, predictions, tensorboard runs and the run history are written",
    )
    parser.add_argument(
        "--workers",
        help="The number of workers to use for the data loaders.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--foundationtype",
        type=str,
        default='',
        help="",
    )
    parser.add_argument(
        "--hp",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    main(args)
