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
import yaml

from HERA import HERA
from utils_HERA import *

def set_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def evaluate_model(model, device, loader):
    model.eval()

    probs = np.zeros(len(loader))
    logits = np.zeros(len(loader))
    labels = np.zeros(len(loader), dtype=int)
    slide_ids = []

    with torch.no_grad():
        for batch_idx, (data, label, slide_id, cohort) in enumerate(loader):
            data, label = data.to(device), label.to(device).float()
            slide_id = slide_id[0]

            logit, Y_prob, _, _ = model(data)
            logit = logit.squeeze(dim=1)  
            Y_prob = Y_prob.squeeze(dim=1)

            probs[batch_idx] = Y_prob.cpu().item()
            logits[batch_idx] = logit.cpu().item()
            labels[batch_idx] = label.cpu().item()
            slide_ids.append(slide_id)

    auc = roc_auc_score(labels, probs)
    print(f'Eval AUC: {auc:.4f}')

    return auc, probs, logits, labels, slide_ids

def main(args):

    # Set random seed for some degree of reproducibility. See PyTorch docs on this topic for caveats.
    # https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
    set_seed()

    if not torch.cuda.is_available():
        raise Exception(
            "No CUDA device available. Training without one is not feasible."
        )
    start_time = time.time()

    df_test = pd.read_csv(args.manifest_test)
    print(f'Read {args.manifest_test} dataset containing {len(df_test)} samples')
    name_cohort = df_test["Cohort"].iloc[1]
    print("Predicting HERA scores for cohort: ", name_cohort)

    print('Initializing HERA model')

    with open(args.config_file, "r") as file:
        try:
            config = yaml.safe_load(file)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)

    device = torch.device("cuda")

    model = HERA(
        input_feature_size=args.input_feature_size,
        precompression_layer=config["HERA"]["precompression_layer"],
        feature_size_comp=config["HERA"]["feature_size_comp"],
        feature_size_attn=config["HERA"]["feature_size_attn"],
        postcompression_layer=config["HERA"]["postcompression_layer"],
        feature_size_comp_post=config["HERA"]["feature_size_comp_post"],
        dropout=True,
        p_dropout_fc=config["HERA"]["p_dropout_fc"],
        p_dropout_atn=config["HERA"]["p_dropout_atn"],
    ).to(device)
    msg = model.load_state_dict(torch.load(args.checkpoint_HERA_model, map_location=device), strict=True)
    print('HERA checkpoint loading', msg)

    print(f"Finished loading HERA in {(time.time() - start_time):.2f}s")

    for p in model.parameters():
        p.requires_grad = False
    
    model.eval()

    dataset = FeatureBags_Binary_ClassificationDataset(df=df_test,
                                    data_dir=args.data_dir,
                                    input_feature_size=args.input_feature_size)

    loader = define_data_sampling_testing(
            val_split = dataset,
            task='classification',
            workers=args.workers
    )

    # Evaluation. 
    eval_auc, probs, logits, labels, slide_ids = evaluate_model(model, device, loader) 

    preds_test = pd.DataFrame(probs, columns=[f'prob_HPV'])
    preds_test['logit'] = logits
    preds_test['label'] = labels
    preds_test['slide_id'] = slide_ids

    # Save into a csv file.
    print(f'Saving predictions...')
    os.makedirs(args.output_dir, exist_ok=True)
    preds_test.to_csv(os.path.join(args.output_dir, f'predictions_{name_cohort}_N{len(df_test)}.csv'), index=False)

    print(f"Finished making HERA predictions in {(time.time() - start_time):.2f}s")

    total_time = time.time() - start_time
    avg_time_per_slide = total_time / len(df_test)
    print(f"Average prediction time per slide: {avg_time_per_slide:.2f}s")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Testing script")
    parser.add_argument(
        "--manifest_test",
        type=str,
        help="CSV file of test set listing all work_ids, slides, cohorts, labels",
    )
    parser.add_argument(
        "--checkpoint_HERA_model",
        type=str,
        help="path to HERA model checkpoint",
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
        "--config_file",
        help="Path to config.yaml file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="predictions",
        help="Directory to write the predictions CSV to",
    )
    parser.add_argument(
        "--workers",
        help="The number of workers to use for the data loaders.",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    main(args)
