import os
import random
import subprocess
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

from sklearn.metrics import roc_auc_score

from HERMES import HERMES
from utils_HERMES import *

def set_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def evaluate_model(epoch, model, device, loader, n_classes, writer, loss_fn, class_names):
    model.eval()
    eval_loss = 0.

    probs = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader), dtype=int)
    slide_ids = []

    with torch.no_grad():
        for batch_idx, (data, label, slide_id, cohort) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            slide_id = slide_id[0]

            logits, Y_prob, _, _ = model(data)  # Assume Y_prob is softmax probs
            loss = loss_fn(logits, label)
            eval_loss += loss.item()

            #.squeeze() after probs?
            probs[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.cpu().item()
            slide_ids.append(slide_id)

    eval_loss /= len(loader)

    # Compute AUC per class
    auc_per_class = {}
    for i in range(n_classes):
        binary_labels = (labels == i).astype(int)
        try:
            auc = roc_auc_score(binary_labels, probs[:, i])
        except ValueError:
            auc = float('nan')
        auc_per_class[class_names[i]] = auc

    # Compute macro average
    valid_aucs = [auc for auc in auc_per_class.values() if not np.isnan(auc)]
    macro_auc = np.mean(valid_aucs)

    # Logging
    print(f'Eval epoch: {epoch}, loss: {eval_loss}, macro AUC: {macro_auc:.4f}')
    for class_name, auc in auc_per_class.items():
        print(f' - {class_name} vs rest AUC: {auc:.4f}')
        writer.add_scalar(f"auc/eval/{class_name}", auc, epoch)

    writer.add_scalar("Loss/eval", eval_loss, epoch)
    writer.add_scalar("auc/eval/macro", macro_auc, epoch)

    return eval_loss, macro_auc, valid_aucs, probs, labels, slide_ids

def train_one_epoch(epoch, model, device, train_loader, optimizer, n_classes, writer, loss_fn, class_names):
    
    model.train()
    epoch_start_time = time.time()
    train_loss = 0.

    probs = np.zeros((len(train_loader), n_classes))
    labels = np.zeros(len(train_loader))
    slide_ids = []

    batch_start_time = time.time()
    for batch_idx, (data, label, slide_id, cohort) in enumerate(train_loader):

        data_load_duration = time.time() - batch_start_time

        data, label = data.to(device), label.to(device)

        logit, Y_prob, _, _ = model(data)

        loss = loss_fn(logit, label)
        train_loss += loss.item()

        probs[batch_idx] = Y_prob.detach().cpu().numpy()
        labels[batch_idx] = label.cpu().item()
        slide_ids.append(slide_id)

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

    # Compute AUC per class
    auc_per_class = {}
    for i in range(n_classes):
        binary_labels = (labels == i).astype(int)
        try:
            auc = roc_auc_score(binary_labels, probs[:, i])
        except ValueError:
            auc = float('nan')
        auc_per_class[class_names[i]] = auc

    # Compute macro average
    valid_aucs = [auc for auc in auc_per_class.values() if not np.isnan(auc)]
    macro_auc = np.mean(valid_aucs)

    # Logging
    print(f'Train epoch: {epoch}, loss: {train_loss}, macro AUC: {macro_auc:.4f}')
    for class_name, auc in auc_per_class.items():
        print(f' - {class_name} vs rest AUC: {auc:.4f}')
        writer.add_scalar(f"auc/train/{class_name}", auc, epoch)

    writer.add_scalar("duration/epoch", epoch_duration, epoch)
    writer.add_scalar("LR", get_lr(optimizer), epoch)
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("auc/train", macro_auc, epoch)

    return probs, labels, slide_ids

def run_train_eval_loop(train_loader, val_loader, loss_fn, hparams, n_classes, run_id, foundation_model, hpset, class_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(output_dir, "runs", run_id))
    
    device = torch.device("cuda")

    model = HERMES(
        input_feature_size=hparams["input_feature_size"],
        precompression_layer=hparams["precompression_layer"],
        feature_size_comp=hparams["feature_size_comp"],
        feature_size_attn=hparams["feature_size_attn"],
        postcompression_layer=hparams["postcompression_layer"],
        feature_size_comp_post=hparams["feature_size_comp_post"],
        dropout=True,
        p_dropout_fc=hparams["p_dropout_fc"],
        p_dropout_atn=hparams["p_dropout_atn"],
        n_classes=n_classes,
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

    monitor_tracker = MonitorBestModelEarlyStopping_3class_final(
        patience=hparams["earlystop_patience"],
        min_epochs=hparams["earlystop_min_epochs"],
        saving_checkpoint=True,
        foundation_model=foundation_model,
        hpset=hpset,
        output_dir=output_dir
    )

    for epoch in range(hparams["max_epochs"]):

        probs_train, labels_train, slide_ids_train = train_one_epoch(epoch, model, device, train_loader, optimizer, n_classes, writer, loss_fn, class_names)

        preds_train = pd.DataFrame(probs_train, columns=[f'prob_{name}' for name in class_names])
        preds_train['label'] = labels_train
        preds_train['slide_id'] = slide_ids_train

        # Evaluation. 
        eval_loss, eval_auc, valid_aucs, probs, labels, slide_ids = evaluate_model(epoch, model, device, val_loader, n_classes, writer, loss_fn, class_names) 

        preds_intval = pd.DataFrame(probs, columns=[f'prob_{name}' for name in class_names])
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

    with open(os.path.join(output_dir, 'run_history_HERMES.txt'), 'a') as filehandle:
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

    n_classes = 3

    class_names = ["HPVpos", "HPVneg/p53wt", "HPVneg/p53abn"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    print("class_to_idx:", class_to_idx)

    try:
        training_set = df_train
        validation_set = df_val
    except:
        raise Exception(
            f"Column does not exist in {args.manifest}"
        )

    train_split = FeatureBags_3Class_ClassificationDataset(df=training_set,
                                    data_dir=args.data_dir,
                                    input_feature_size=args.input_feature_size,
                                    class_to_idx=class_to_idx)

    val_split = FeatureBags_3Class_ClassificationDataset(df=validation_set,
                                    data_dir=args.data_dir, 
                                    input_feature_size=args.input_feature_size,
                                    class_to_idx=class_to_idx)
    

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
        # Preprocessing settings. This should be changed with the dataset called accordingly.
        # Storing values here for readibility.
        n_classes=n_classes, 

        input_feature_size=args.input_feature_size,
        features_extraction=os.path.dirname(args.data_dir),

        # Settings that be changed in the loop:
        # Training.
        sampling_method="random",
        class_weighting=False,
        max_epochs=30,
        earlystop_patience=30, #whenever loss is not decreasing, keep training for minimum earlystop_patience patience
        earlystop_min_epochs=30, #minimum training before early stopping works

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

        { # hp0
            **base_hparams,
        },
        { # hp1
            **base_hparams,
            'feature_size_attn': 128,
            'initial_lr':0.00004,
            'milestones':"5, 12, 20",
            'gamma_lr':0.35,
            'weight_decay':0.00005,
            'max_epochs':100,
        },  
        { # hp2
             **base_hparams,
             'class_weighting': True,
             'max_epochs':100,
        },  
        { # hp3
            **base_hparams,
            'initial_lr':0.00004,
            'milestones':"10, 15, 20",
            'gamma_lr':0.5,
            'weight_decay':0.00005,
        }, 
        { # hp4
             **base_hparams,
             'sampling_method': "balanced",
        },  
        { # hp5
            **base_hparams,
            'p_dropout_fc':0.50,
            'p_dropout_atn':0.50,
            'initial_lr':0.00004,
            'milestones':"10, 15, 20",
            'gamma_lr':0.1,
            'weight_decay':0.00005,
        }, 
        { # hp6
            **base_hparams,
            'feature_size_comp_post':256,
        },
        { # hp7   
             **base_hparams,
            'hierarchical_loss': True,
            'hierarchical_loss_version': "hpv-based",
            'label_smoothing':False,
        },
        { # hp8  
             **base_hparams,
            'hierarchical_loss': True,
            'hierarchical_loss_version': "hpv_p53-based",
            'label_smoothing':False,
        },
        { # hp9   
             **base_hparams,
            'hierarchical_loss': True,
            'hierarchical_loss_version': "p53-based",
            'label_smoothing':False,
        },
        { # hp10
            **base_hparams,
            'label_smoothing':True,
            'class_weighting': True,
        },
        { # hp11 --> now with only one net and one classifier with 3 nodes
             **base_hparams,
             'class_weighting': True,
        },  
        { # hp12 --> start here with optimizing hpset 2. we start with some more dropout.
            **base_hparams,
            'class_weighting': True,
            'p_dropout_fc':0.35,
            'p_dropout_atn':0.35,
        },  
        { # hp13 --> now we try less time to memorize and less aggresive drops
            **base_hparams,
            'class_weighting': True,
            'milestones':"5, 10, 15",
            'gamma_lr':0.4,
            'weight_decay':0.0001,
        },  
        { # hp14 
             **base_hparams,
             'class_weighting': True,
             'initial_lr':0.000003,
             'weight_decay':0.00003,
        }, 
        { # hp15
             **base_hparams,
             'class_weighting': True,
             'gamma_lr':0.2,
             'weight_decay':0.00003,
        }, 
        { # hp16
             **base_hparams,
             'class_weighting': True,
             'initial_lr':0.00006,
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
            workers=args.workers,
    )

    device = torch.device("cuda")   

    if hps["class_weighting"]:

        all_labels = torch.tensor(
            [train_split[i][1] for i in range(len(train_split))]
        )

        class_counts = torch.bincount(all_labels, minlength=3).float()
        weights = class_counts.sum() / class_counts
        weights = weights / weights.sum() * len(weights)
        weights = weights.to(device)
        print("weights:", weights)

        lossfunction = nn.CrossEntropyLoss(weight=weights)

    else:
        lossfunction = nn.CrossEntropyLoss()
    

    run_train_eval_loop(
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn = lossfunction,
            hparams=hps,
            n_classes=n_classes,
            run_id=run_id,
            foundation_model = args.foundationtype, 
            hpset = args.hp,
            output_dir = args.output_dir,
            class_names = class_names        
    )
    print("Finished training.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--manifest_training",
        type=str,
        help="CSV file listing all slides, their labels, and which split (train/test/val) they belong to.",
    )
    parser.add_argument(
        "--manifest_intval",
        type=str,
        help="CSV file listing all slides, their labels, and which split (train/test/val) they belong to.",
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
        default="outputs/HERMES",
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
