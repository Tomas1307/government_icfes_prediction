import mlflow
import mlflow.tensorflow
from sklearn.model_selection import KFold
import tensorflow as tf
from convolutional import ConvRegressor
import itertools
import numpy as np

def run_gridsearch_with_mlflow(df, target_col, param_grid, n_splits=3, experiment_name="convregressor-grid"):
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name)

    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))

    for combo in combinations:
        params = dict(zip(keys, combo))

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        maes = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            df_train, df_val = df.iloc[train_idx], df.iloc[val_idx]
            model = ConvRegressor(
                embedding_dim=params['embedding_dim'],
                conv_filters=params['conv_filters'],
                dense_units=params['dense_units']
            )
            model.build_model(df_train, target_col)
            model.train(df_train, target_col, epochs=params['epochs'], batch_size=params['batch_size'])

            preds = model.predict(df_val)
            y_true = df_val[target_col].values
            mae = tf.keras.metrics.mean_absolute_error(y_true, preds.squeeze()).numpy()
            maes.append(mae)

        avg_mae = np.mean(maes)

        with mlflow.start_run():
            for k, v in params.items():
                mlflow.log_param(k, v)
            mlflow.log_metric("avg_mae", avg_mae)
            mlflow.tensorflow.log_model(model.model, "model")
            print(f"Logged run with avg MAE: {avg_mae:.4f}")
