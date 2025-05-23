import os
import shutil
import numpy as np
import mlflow
import mlflow.pytorch
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from convolutional import ConvRegressor, ConvClassifier
import torch
import matplotlib.pyplot as plt

def plot_loss_curve(history, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    if any(v is not None for v in history["val_loss"]):
        plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def run_gridsearch_with_mlflow(
    df,
    target_col,
    param_grid,
    task_type='regression',
    experiment_name="gridsearch",
    save_dir="checkpoints/",
    return_best_model=True,
    test_size=0.2,
    val_frac=0.1
):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

    os.makedirs(save_dir, exist_ok=True)
    best_score = float("inf") if task_type == 'regression' else float("-inf")
    best_model = None
    best_path = None

    from itertools import product
    keys = list(param_grid.keys())
    combinations = list(product(*[param_grid[k] for k in keys]))

    for combo in combinations:
        params = dict(zip(keys, combo))

        # Split data
        df_trainval, df_test = train_test_split(df, test_size=test_size, random_state=42)
        df_train, df_val = train_test_split(df_trainval, test_size=val_frac, random_state=42)
        # Eliminar variables derivadas para evitar data leakage
        if target_col == 'punt_matematicas':
            df_train = df_train.drop(columns=['eco'], errors='ignore')
            df_val = df_val.drop(columns=['eco'], errors='ignore')
        elif target_col == 'eco':
            df_train = df_train.drop(columns=['punt_matematicas'], errors='ignore')
            df_val = df_val.drop(columns=['punt_matematicas'], errors='ignore')


        # Init model
        model_class = ConvRegressor if task_type == 'regression' else ConvClassifier
        model = model_class(
            embedding_dim=params['embedding_dim'],
            conv_filters=params['conv_filters'],
            dense_units=params['dense_units']
        )
        model.build_model(df_train.copy(), target_col)

        # Train
        model.train(
            target_col=target_col,
            val_df=df_val.copy(),
            early_stopping=True,
            patience=params.get("patience", 5),
            epochs=params['epochs'],
            batch_size=params['batch_size']
        )

        features_to_exclude = [target_col]

        if target_col == 'punt_matematicas':
            features_to_exclude.append('eco')
        elif target_col == 'eco':
            features_to_exclude.append('punt_matematicas')

        X_val = df_val.drop(columns=features_to_exclude, errors='ignore')

        y_true = df_val[target_col].values
        y_pred = model.predict(X_val).squeeze()

        with mlflow.start_run():
            for k, v in params.items():
                mlflow.log_param(k, v)

            if task_type == 'regression':
                mse = mean_squared_error(y_true, y_pred)
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, y_pred)

                mlflow.log_metric("mse", mse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)

                score = mse  
            else:
                y_pred_bin = (y_pred > 0.5).astype(int)
                recall = recall_score(y_true, y_pred_bin)
                acc = accuracy_score(y_true, y_pred_bin)
                precision = precision_score(y_true, y_pred_bin)
                f1 = f1_score(y_true, y_pred_bin)
                try:
                    roc_auc = roc_auc_score(y_true, y_pred)
                except:
                    roc_auc = 0.0

                mlflow.log_metric("recall", recall)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("f1", f1)
                mlflow.log_metric("roc_auc", roc_auc)

                score = recall  

            is_better = score < best_score if task_type == 'regression' else score > best_score
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            model_path = os.path.join(save_dir, f"model_{task_type}_{timestamp}.pt")
            torch.save(model.model.state_dict(), model_path)

            loss_curve_path = os.path.join(save_dir, f"loss_curve_{task_type}_{timestamp}.png")
            plot_loss_curve(model.history, loss_curve_path)

            mlflow.pytorch.log_model(model.model, "model")
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(loss_curve_path)

            if is_better:
                best_score = score
                best_model = model
                best_path = model_path

    if return_best_model:
        print(f"\n Mejor modelo guardado en: {best_path}")
        return best_model
    return None
