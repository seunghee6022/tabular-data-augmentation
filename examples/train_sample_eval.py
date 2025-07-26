# Execute only once!
import os
import sys
import torch
import time
import random
import logging
import json

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

from be_great import GReaT
from utils import set_logging_level

# Setup
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Logging & seed
logger = set_logging_level(logging.INFO)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def evaluate_downstream_task(real_data, synthetic_data, target_col, model_type="LR"):
    # Split into X and y
    X_train = synthetic_data.drop(columns=[target_col])
    y_train = synthetic_data[target_col]
    X_test = real_data.drop(columns=[target_col])
    y_test = real_data[target_col]

    # Handle categorical features
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols_test = [col for col in cat_cols if col in X_test.columns]
    print(f"cat_cols>{cat_cols} cat_cols_test>{cat_cols_test}")


    if cat_cols:
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
        X_test = pd.get_dummies(X_test, columns=cat_cols_test, drop_first=True)

        # Align columns to ensure same shape
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

    # Encode y if it's a string
    if y_train.dtype == 'object':
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    
    if model_type == "LR":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "DT":
        model = DecisionTreeClassifier()
    elif model_type == "RF":
        model = RandomForestClassifier()
    else:
        raise ValueError("Invalid model_type")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # if y_test.nunique() > 10:
    if pd.Series(y_test).nunique() > 30:
        return {
            "mse": mean_squared_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        }
    else:
        return {
            "accuracy": accuracy_score(y_test, y_pred)
        }

def evaluate_discriminator(real_data, synthetic_data):
    real_data = real_data.copy()
    synthetic_data = synthetic_data.copy()

    # Combine real and synthetic data
    X_combined = pd.concat([real_data, synthetic_data], ignore_index=True)
    y = np.array([0] * len(real_data) + [1] * len(synthetic_data))  # 0 = real, 1 = synthetic

    # Encode categorical variables
    cat_cols = X_combined.select_dtypes(include=['object', 'category']).columns
    X_encoded = pd.get_dummies(X_combined, columns=cat_cols, drop_first=True)

    # Fill NaNs introduced by mismatched categories
    X_encoded = X_encoded.fillna(0)

    # Fit logistic regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_encoded, y)
    y_pred = clf.predict(X_encoded)

    return {"discriminator_accuracy": accuracy_score(y, y_pred)}

# def compute_dcr_score(real_data, synthetic_data):
#     # Combine data for consistent encoding
#     combined = pd.concat([real_data, synthetic_data], ignore_index=True)

#     # One-hot encode categorical columns
#     cat_cols = combined.select_dtypes(include=['object', 'category']).columns
#     combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

#     # Split back into real and synthetic after encoding
#     X_real = combined_encoded.iloc[:len(real_data), :]
#     X_syn = combined_encoded.iloc[len(real_data):, :]

#     # Fill NaNs introduced by mismatched categories
#     X_real_encoded = X_real.fillna(0)

#     # Scale numeric features
#     scaler = MinMaxScaler()
#     # X_real_scaled = scaler.fit_transform(X_real)
#     X_real_scaled = scaler.fit_transform(X_real_encoded)
#     X_syn_scaled = scaler.transform(X_syn)

#     # Nearest neighbors for DCR
#     nn = NearestNeighbors(n_neighbors=1, metric='manhattan')
#     nn.fit(X_real_scaled)
#     distances, _ = nn.kneighbors(X_syn_scaled)

#     return {
#         "avg_dcr_l1": float(distances.mean()),
#         "std_dcr_l1": float(distances.std())
#     }

from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import pandas as pd

def compute_dcr_score(real_data, synthetic_data):
    # Combine data for consistent encoding
    combined = pd.concat([real_data, synthetic_data], ignore_index=True)

    # One-hot encode categorical columns
    cat_cols = combined.select_dtypes(include=['object', 'category']).columns
    combined_encoded = pd.get_dummies(combined, columns=cat_cols, drop_first=True)

    # Split back into real and synthetic
    X_real_encoded = combined_encoded.iloc[:len(real_data), :].copy()
    X_syn_encoded = combined_encoded.iloc[len(real_data):, :].copy()

    # Fill NaNs introduced by mismatched categories
    X_real_encoded = X_real_encoded.fillna(0)

    # Make sure both have same columns (handle if some category appears only in real or synthetic)
    X_syn_encoded = X_syn_encoded.reindex(columns=X_real_encoded.columns, fill_value=0)

    # Scale numeric features to [0, 1]
    scaler = MinMaxScaler()
    X_real_scaled = scaler.fit_transform(X_real_encoded)
    X_syn_scaled = scaler.transform(X_syn_encoded)

    # Nearest neighbor distances (Manhattan distance = L1)
    nn = NearestNeighbors(n_neighbors=1, metric='manhattan')
    nn.fit(X_real_scaled)
    distances, _ = nn.kneighbors(X_syn_scaled)

    return {
        "avg_dcr_l1": float(distances.mean()),
        "std_dcr_l1": float(distances.std())
    }


# Config
epochs = 50
batch_size = 16
sample_num = "train" #5000

dataset_configs = {
    "california": epochs,
    "insurance": epochs,
    "adult": epochs,
    "heloc": epochs,
    "covtype": epochs,
    "intrusion": epochs,
}

column_dict = {
    "iris": ["sepal length", "sepal width", "petal length", "petal width", "target"],
    "california": [],
    "insurance": [],
    "adult": [],
    "heloc": [],
    "covtype": [],
    "intrusion": []
}

# Loop through datasets
for dataset_name, n_epochs in dataset_configs.items():
    logger.info(f"--- Starting training for dataset: {dataset_name} ---")
    output_dir = f"sample{sample_num}_epochs{n_epochs}_seed{SEED}_earlystopping"
    # output_dir = f"sample{sample_num}_epochs{n_epochs}_seed{SEED}"
    os.makedirs(f"outputs/{output_dir}", exist_ok=True)

    # Load dataset
    if dataset_name == "california":
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
        # data = datasets.fetch_kddcup99(as_frame=True, percent10=True).frame
        # Convert all byte strings to regular strings
        # data = data.applymap(lambda x: x.decode() if isinstance(x, bytes) else x)
        # data.to_csv("dataset/intrusion.csv", index=False)
        data = pd.read_csv("dataset/intrusion.csv")

    else:
        logger.warning(f"Dataset {dataset_name} not found.")
        continue

    if column_dict[dataset_name]:
        data.columns = column_dict[dataset_name]

    # Identify target column
    target_col = (
        "target" if "target" in data.columns else
        "labels" if "labels" in data.columns else
        "class" if "class" in data.columns else
        "charges" if "charges" in data.columns else
        "MedHouseVal" if "MedHouseVal" in data.columns else
        "RiskPerformance" if "RiskPerformance" in data.columns else
        "Cover_Type" if "Cover_Type" in data.columns else
        None
    )
    if not target_col:
        logger.warning(f"Unknown target column for dataset: {dataset_name}")
        continue
    else:
        print(f"target col:{target_col}")

    # Stratify
    stratify_col = data[target_col] if target_col in data.columns else None

    # Split
    train_data, temp_data = train_test_split(
        # data, test_size=0.2, random_state=SEED, stratify=stratify_col
        data, test_size=0.2, random_state=SEED,
    )
    val_data, test_data = train_test_split(
        # temp_data, test_size=0.5, random_state=SEED, stratify=temp_data[target_col]
        temp_data, test_size=0.5, random_state=SEED,
    )
    # train_data = train_data.head(sample_num)
    logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}, Test size: {len(test_data)}")

    # Train GReaT
    experiment_dir = f"outputs/{output_dir}/trainer_{dataset_name}"
    
    # Step 1: Select early stopping settings based on dataset
    if dataset_name in ["california", "insurance"]:
        early_stopping_metric = "mse" 
        early_stopping_mode = "min"
    else:
        early_stopping_metric = "accuracy" 
        early_stopping_mode = "max"

    # Step 2: Initialize GReaT
    great = GReaT(
        llm="distilgpt2",
        epochs=n_epochs,
        experiment_dir=experiment_dir,
        batch_size=batch_size,
        train_hyperparameters={
            "save_steps": 5000,
            "logging_steps": 1000,
            "save_total_limit": 1,
        }
    )

    # Step 3: Train with early stopping
    start_time = time.time()

    trainer = great.fit(
        data=train_data,
        eval_data=val_data,
        early_stopping_patience=3,
        early_stopping_metric=early_stopping_metric,
        early_stopping_mode=early_stopping_mode,
    )

    elapsed = time.time() - start_time

    # Step 4: Save model
    save_path = f"outputs/{output_dir}/model_{dataset_name}"
    great.save(save_path)

    # Step 5: Sample synthetic data
    samples = great.sample(1000, k=50, device="cuda:0")
    samples.to_csv(f"outputs/{output_dir}/{dataset_name}_samples.csv")

    # Step 4-2: load the samples
    # samples = pd.read_csv(f"outputs/{output_dir}/{dataset_name}_samples.csv")

    # Step 6: Log training
    training_log = {
        "dataset": dataset_name,
        "epochs": n_epochs,
        "batch_size": batch_size,
        "seed": SEED,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "training_time_sec": round(elapsed, 2),
        "model_save_path": save_path,
        "samples_path": f"outputs/{output_dir}/{dataset_name}_samples.csv",
        "start_time": start_time,
        "end_time": time.time()
    }


    if hasattr(trainer, "metrics"):
        training_log["metrics"] = trainer.metrics

    # Evaluation Metrics
    training_log["downstream_performance"] = {}
    for model_type in ["LR", "DT", "RF"]:
        training_log["downstream_performance"][model_type] = {
            "val": evaluate_downstream_task(val_data, samples, target_col, model_type),
            "test": evaluate_downstream_task(test_data, samples, target_col, model_type)
        }
    print(f"downstream_performance --- training_log: {training_log}")

    training_log["discriminator_eval"] = {
        "val": evaluate_discriminator(val_data.drop(columns=[target_col]), samples.drop(columns=[target_col])),
        "test": evaluate_discriminator(test_data.drop(columns=[target_col]), samples.drop(columns=[target_col]))
    }
    print(f"discriminator_eval --- training_log: {training_log}")


    training_log["distribution_alignment"] = {
        "real-real": compute_dcr_score(val_data.drop(columns=[target_col]), val_data.drop(columns=[target_col])),
        "val": compute_dcr_score(val_data.drop(columns=[target_col]), samples.drop(columns=[target_col])),
        "test": compute_dcr_score(test_data.drop(columns=[target_col]), samples.drop(columns=[target_col]))
    }
    print(f"distribution_alignment --- training_log: {training_log}")

    # Save training log
    with open(f"outputs/{output_dir}/{dataset_name}_training_log.json", "w") as f:
        json.dump(training_log, f, indent=4)

    logger.info(f"Finished training and evaluation for dataset: {dataset_name}")
    logger.info(f"Training log saved to outputs/{output_dir}/{dataset_name}_training_log.json")