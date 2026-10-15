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

from HERMES import HERMES
from utils_HERMES import *

def set_seed():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def evaluate_model(model, device, loader, n_classes, class_names):
    model.eval()

    probs = np.zeros((len(loader), n_classes))
    logits = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader), dtype=int)
    slide_ids = []

    with torch.no_grad():
        for batch_idx, (data, label, slide_id, cohort) in enumerate(loader):
            data, label = data.to(device), label.to(device)
            slide_id = slide_id[0]

            logit, Y_prob, _, _ = model(data)

            probs[batch_idx] = Y_prob.cpu().numpy()
            logits[batch_idx] = logit.cpu().numpy()
            labels[batch_idx] = label.cpu().item()
            slide_ids.append(slide_id)

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
    print(f'Macro AUC: {macro_auc:.4f}, HPVpos: {valid_aucs[0]:.4f}, HPV-p53wt: {valid_aucs[1]:.4f}, HPV-p53abn: {valid_aucs[2]:.4f}')

    return macro_auc, valid_aucs, probs, logits, labels, slide_ids

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
    print("Predicting HERMES scores for cohort: ", name_cohort)

    n_classes = 3

    class_names = ["HPVpos", "HPVneg/p53wt", "HPVneg/p53abn"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    print("class_to_idx:", class_to_idx)

    print('Initializing HERMES model')

    with open(args.config_file, "r") as file:
        try:
            config = yaml.safe_load(file)
            print(config)
        except yaml.YAMLError as exc:
            print(exc)

    device = torch.device("cuda")

    model = HERMES(
        input_feature_size=args.input_feature_size,
        precompression_layer=config["HERMES"]["precompression_layer"],
        feature_size_comp=config["HERMES"]["feature_size_comp"],
        feature_size_attn=config["HERMES"]["feature_size_attn"],
        postcompression_layer=config["HERMES"]["postcompression_layer"],
        feature_size_comp_post=config["HERMES"]["feature_size_comp_post"],
        dropout=True,
        p_dropout_fc=config["HERMES"]["p_dropout_fc"],
        p_dropout_atn=config["HERMES"]["p_dropout_atn"],
    ).to(device)
    msg = model.load_state_dict(torch.load(args.checkpoint_HERMES_model, map_location=device), strict=True)
    print('HERMES checkpoint loading', msg)

    print(f"Finished loading HERMES in {(time.time() - start_time):.2f}s")

    for p in model.parameters():
        p.requires_grad = False
    
    model.eval()

    dataset = FeatureBags_3Class_ClassificationDataset(df=df_test,
                                    data_dir=args.data_dir,
                                    input_feature_size=args.input_feature_size,
                                    class_to_idx=class_to_idx)

    loader = define_data_sampling_testing(
            val_split = dataset,
            task='classification',
            workers=args.workers
    )

    # Evaluation. 
    eval_macro_auc, eval_all_aucs, probs, logits, labels, slide_ids = evaluate_model(model, device, loader, n_classes, class_names) 

    preds_test = pd.DataFrame(probs, columns=[f'prob_{name}' for name in class_names])
    preds_test['logit_HPV'] = logits[:, 0]
    preds_test['logit_HPV_p53wt'] = logits[:, 1]
    preds_test['logit_HPV_p53abn'] = logits[:, 2]
    preds_test['label'] = labels
    preds_test['slide_id'] = slide_ids

    # Save into a csv file.
    print(f'Saving predictions...')
    os.makedirs(args.output_dir, exist_ok=True)
    preds_test.to_csv(os.path.join(args.output_dir, f'predictions_{name_cohort}_N{len(df_test)}.csv'), index=False)

    print(f"Finished making HERMES predictions in {(time.time() - start_time):.2f}s")

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
        "--checkpoint_HERMES_model",
        type=str,
        help="path to HERMES model checkpoint",
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
