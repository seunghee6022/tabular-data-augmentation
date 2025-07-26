import argparse
import itertools
import os
import sys
import torch
import time
import random
import logging
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

from be_great import GReaT
from utils import set_logging_level
import json

def main(args):
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    # Set global logging
    logger = set_logging_level(logging.INFO)

    # Set seed for reproducibility
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    epochs = args.epochs

    column_dict = {
        "iris": ["sepal length", "sepal width", "petal length", "petal width", "target"],
        "california": [],
        "insurance": [],
        "adult": [],
        "heloc": [],
        "covtype": [],
        "intrusion": []
    }
    # Hyperparameter grid
    param_grid = {
        # "epochs": args.epochs_grid,
        "batch_size": args.batch_size_grid,
        "llm": args.llm_grid,
    }
    grid = list(itertools.product(*param_grid.values()))

    for dataset_name in args.datasets:
        # for epochs, batch_size, llm in grid:
        for batch_size, llm in grid:
            logger.info(f"--- Training {dataset_name} | epochs={epochs}, batch_size={batch_size}, llm={llm} ---")
            output_dir = f"train_data_epochs{epochs}_batch_size_{batch_size}_llm_{llm}_seed{SEED}_lora"

            # Load dataset
            if dataset_name == "iris":
                data = datasets.load_iris(as_frame=True).frame
            elif dataset_name == "california":
                data = datasets.fetch_california_housing(as_frame=True).frame
            elif dataset_name == "insurance":
                data = pd.read_csv("dataset/Insurance_compressed.csv") 
            elif dataset_name == "adult":
                data = datasets.fetch_openml("adult", version=2, as_frame=True).frame
            elif dataset_name == "heloc":
                data = pd.read_csv("dataset/heloc_dataset_v1.csv")
            elif dataset_name == "covtype":
                data = datasets.fetch_covtype(as_frame=True).frame
            elif dataset_name == "intrusion":
                data = pd.read_csv("dataset/intrusion.csv")
            else:
                logger.warning(f"Dataset {dataset_name} not found.")
                continue

            # Rename columns if specified
            if column_dict[dataset_name]:
                data.columns = column_dict[dataset_name]

            # Encode string labels to integers for classification datasets and set stratify_col
            stratify_col = None
            if dataset_name not in ["california", "insurance"]: # Only classification datasets
                label_col = None
                if "target" in data.columns:
                    label_col = "target"
                elif dataset_name == "adult" and "class" in data.columns:
                    label_col = "class"
                elif dataset_name == "heloc" and "RiskPerformance" in data.columns:
                    label_col = "RiskPerformance"
                elif dataset_name == "covtype" and "Cover_Type" in data.columns:
                    label_col = "Cover_Type"
                elif dataset_name == "intrusion" and "labels" in data.columns:
                    label_col = "labels"
                if label_col is not None:
                    if data[label_col].dtype == object:
                        data[label_col], _ = pd.factorize(data[label_col])
                    stratify_col = data[label_col]
            else:
                # For regression, stratify_col logic as before
                if "target" in data.columns:
                    stratify_col = data["target"]
                elif dataset_name == "insurance" and "charges" in data.columns:
                    stratify_col = data["charges"]
                else:
                    stratify_col = None

            # Split into train/val/test
            train_data, temp_data = train_test_split(
                data, test_size=0.2, random_state=SEED, stratify=stratify_col if stratify_col is not None else None
            )
            val_data, test_data = train_test_split(
                temp_data, test_size=0.5, random_state=SEED, stratify=stratify_col[temp_data.index] if stratify_col is not None else None
            )
            logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")

            # Setup GReaT
            experiment_dir = f"outputs/{output_dir}/trainer_{dataset_name}"
            great = GReaT(
                llm=llm,
                epochs=epochs,
                efficient_finetuning= "lora",
                experiment_dir=experiment_dir,
                batch_size=batch_size,  
            )

            start_time = time.time()

            # Set early stopping metric based on dataset type
            if dataset_name in ["california", "insurance"]:
                early_stopping_metric = "mse"
                early_stopping_mode = "min"
            else:
                early_stopping_metric = "accuracy"
                early_stopping_mode = "max"

            # Train
            if isinstance(train_data, pd.DataFrame):
                trainer = great.fit(
                    train_data,
                    eval_data=val_data,
                    early_stopping_patience=args.early_stopping_patience,
                    early_stopping_metric=early_stopping_metric,
                    early_stopping_mode=early_stopping_mode
                )
            else:
                column_names = column_dict[dataset_name]
                trainer = great.fit(
                    train_data,
                    eval_data=val_data,
                    column_names=column_names,
                    early_stopping_patience=args.early_stopping_patience,
                    early_stopping_metric=early_stopping_metric,
                    early_stopping_mode=early_stopping_mode
                )

            elapsed = time.time() - start_time

            # Save model
            save_path = f"outputs/{output_dir}/model_{dataset_name}"
            great.save(save_path)

            # Save logging data as JSON
            log_data = {
                "status": "finished",
                "dataset": dataset_name,
                "training_time_sec": elapsed,
                "model_path": save_path,
                "epochs": epochs,
                "batch_size": batch_size,
                "llm": llm,
                "seed": SEED
            }
            log_json_path = f"outputs/{output_dir}/train_log_{dataset_name}.json"
            with open(log_json_path, "w") as f:
                json.dump(log_data, f, indent=2)

            great = GReaT.load_from_dir(save_path)
            samples = great.sample(args.sample_size, k=50, device=args.device)
            samples.to_csv(f"outputs/{output_dir}/{dataset_name}_samples.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search for GReaT hyperparameters")
    parser.add_argument("--datasets", nargs="+", default=[ "covtype", "intrusion", "insurance", "heloc", "adult", "california"]) #["california", "insurance", "adult", "heloc", "covtype", "intrusion"]
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sample_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size_grid", nargs="+", type=int, default=[8])
    parser.add_argument("--llm_grid", nargs="+", type=str, default=["distilgpt2"])
    parser.add_argument("--eval_steps", type=int, default=1000, help="Number of steps between evaluations.")
    parser.add_argument("--evaluation_strategy", type=str, default="steps", help="Evaluation strategy: 'steps' or 'epoch'.")
    parser.add_argument("--save_strategy", type=str, default="steps", help="Save strategy: 'steps' or 'epoch'.")
    parser.add_argument("--save_steps", type=int, default=5000, help="Number of steps between model saves.")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Number of evaluations with no improvement to wait before stopping early.")
    parser.add_argument("--early_stopping_metric", type=str, default="accuracy", help="Metric to monitor for early stopping.")
    parser.add_argument("--early_stopping_mode", type=str, default="max", help="Whether higher metric is better ('max') or lower ('min').")
    args = parser.parse_args()
    main(args)
