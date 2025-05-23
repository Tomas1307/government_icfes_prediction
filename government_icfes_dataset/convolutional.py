import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from label_encoder_unk import LabelEncoderWithUnknown
from tqdm import tqdm
import time


class ConvRegressorModel(nn.Module):
    def __init__(self, cat_dims, num_features, embedding_dim=4, conv_filters=[16, 32], dense_units=64):
        super(ConvRegressorModel, self).__init__()
        self.embedding_dim = embedding_dim
        self.conv_filters = conv_filters
        self.dense_units = dense_units
        
        self.embeddings = nn.ModuleList()
        for vocab_size in cat_dims:
            self.embeddings.append(nn.Embedding(vocab_size + 1, embedding_dim))
        
        self.input_size = (len(cat_dims) * embedding_dim) + num_features
        
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        self.conv_layers.append(nn.Conv1d(in_channels=1, out_channels=conv_filters[0], kernel_size=3, padding=1))
        self.bn_layers.append(nn.BatchNorm1d(conv_filters[0]))
        self.pool_layers.append(nn.MaxPool1d(kernel_size=2))
        self.dropouts.append(nn.Dropout(0.2))
        
        for i in range(1, len(conv_filters)):
            self.conv_layers.append(nn.Conv1d(in_channels=conv_filters[i-1], out_channels=conv_filters[i], kernel_size=3, padding=1))
            self.bn_layers.append(nn.BatchNorm1d(conv_filters[i]))
            self.pool_layers.append(nn.MaxPool1d(kernel_size=2))
            self.dropouts.append(nn.Dropout(0.2))
        
        self.final_conv = nn.Conv1d(in_channels=conv_filters[-1], out_channels=64, kernel_size=3, padding=1)
        self.final_bn = nn.BatchNorm1d(64)
        self.final_dropout = nn.Dropout(0.3)
        
        flattened_size = 64 * (self.input_size // (2 ** len(self.conv_layers)))
        
        self.dense1 = nn.Linear(flattened_size, dense_units)
        self.dropout1 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(dense_units, dense_units // 2)
        self.dropout2 = nn.Dropout(0.2)
        self.output_layer = nn.Linear(dense_units // 2, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, cat_inputs, num_inputs):
        embeddings_output = []
        for i, emb_layer in enumerate(self.embeddings):
            embeddings_output.append(emb_layer(cat_inputs[:, i]).squeeze(1))
        
        x = torch.cat(embeddings_output + [num_inputs], dim=1)
        
        x = x.unsqueeze(1)
        
        for conv, bn, pool, dropout in zip(self.conv_layers, self.bn_layers, self.pool_layers, self.dropouts):
            x = self.relu(bn(conv(x)))
            x = pool(x)
            x = dropout(x)
        
        x = self.relu(self.final_bn(self.final_conv(x)))
        x = self.final_dropout(x)
        
        x = x.flatten(start_dim=1)
        
        x = self.relu(self.dense1(x))
        x = self.dropout1(x)
        x = self.relu(self.dense2(x))
        x = self.dropout2(x)
        
        return self.output_layer(x)


class ConvRegressor:
    def __init__(self, embedding_dim=4, conv_filters=[16, 32], dense_units=64):
        self.embedding_dim = embedding_dim
        self.conv_filters = conv_filters
        self.dense_units = dense_units
        self.encoders = {}
        self.scaler = None
        self.model = None
        self.cat_columns = []
        self.num_columns = []
        self.preprocessed_df = pd.DataFrame()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = {"train_loss": [], "val_loss": []}

    def _identify_columns(self, df, target_col):
        self.cat_columns = df.select_dtypes(include=['object', 'category']).columns.drop(target_col, errors='ignore').tolist()
        self.num_columns = df.select_dtypes(include=[np.number]).columns.drop(target_col, errors='ignore').tolist()

    def _preprocess(self, df):
        df = df.dropna(ignore_index=True)

        for col in self.cat_columns:
            le = LabelEncoderWithUnknown()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le

        self.scaler = StandardScaler()
        df[self.num_columns] = self.scaler.fit_transform(df[self.num_columns])
        return df

    def build_model(self, df, target_col):
        self._identify_columns(df, target_col)
        df = self._preprocess(df)
        self.preprocessed_df = df  

        cat_dims = [df[col].nunique() for col in self.cat_columns]
        num_features = len(self.num_columns)
        
        self.model = ConvRegressorModel(
            cat_dims=cat_dims,
            num_features=num_features,
            embedding_dim=self.embedding_dim,
            conv_filters=self.conv_filters,
            dense_units=self.dense_units
        ).to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        
        return self.model


    def train(self, df=None, target_col=None, val_df=None, early_stopping=False, patience=10, epochs=30, batch_size=16):
        self.history["train_loss"].clear()
        self.history["val_loss"].clear()

        
        if self.model is None:
            raise ValueError("The model has not been built. Call build_model() first.")
        
        if df is None:
            if self.preprocessed_df.empty:
                raise ValueError("No preprocessed DataFrame available. Call build_model() with a DataFrame first.")
            df = self.preprocessed_df

        cat_inputs = torch.tensor(df[self.cat_columns].values, dtype=torch.long).to(self.device)
        num_inputs = torch.tensor(df[self.num_columns].values, dtype=torch.float32).to(self.device)
        targets = torch.tensor(df[target_col].values, dtype=torch.float32).reshape(-1, 1).to(self.device)
        
        dataset = TensorDataset(cat_inputs, num_inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if early_stopping and val_df is not None:
            val_df = val_df.copy()
            for col in self.cat_columns:
                val_df[col] = self.encoders[col].transform(val_df[col].astype(str))
            val_df[self.num_columns] = self.scaler.transform(val_df[self.num_columns])

            val_cat = torch.tensor(val_df[self.cat_columns].values, dtype=torch.long).to(self.device)
            val_num = torch.tensor(val_df[self.num_columns].values, dtype=torch.float32).to(self.device)
            val_targets = torch.tensor(val_df[target_col].values, dtype=torch.float32).reshape(-1, 1).to(self.device)

            early_stopper = EarlyStopping(patience=patience, verbose=True)

        total_features = len(self.cat_columns) + len(self.num_columns)
        print(f" Iniciando entrenamiento en {self.device} con {len(df)} muestras, {total_features} características ({len(self.cat_columns)} categóricas, {len(self.num_columns)} numéricas)")
        
        self.model.train()
        for epoch in range(epochs):
            start_time = time.time()
            running_loss = 0.0
            
            progress_bar = tqdm(dataloader, desc=f"Época {epoch+1}/{epochs}", leave=False)
            
            for cat_batch, num_batch, target_batch in progress_bar:
                self.optimizer.zero_grad()
                outputs = self.model(cat_batch, num_batch)
                loss = self.criterion(outputs, target_batch)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
            
            duration = time.time() - start_time
            epoch_loss = running_loss / len(dataloader)
            self.history["train_loss"].append(epoch_loss)
            log = f" Época {epoch+1}/{epochs} | Train Loss: {epoch_loss:.4f} | Tiempo: {duration:.2f}s"

            if early_stopping and val_df is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(val_cat, val_num)
                    val_loss = self.criterion(val_outputs, val_targets).item()
                self.history["val_loss"].append(val_loss)
                log += f" | Val Loss: {val_loss:.4f}"
                early_stopper(val_loss, self.model)
                if early_stopper.early_stop:
                    print(log)
                    print(f"Early stopping en época {epoch+1}")
                    self.model.load_state_dict(early_stopper.best_model_state)
                    break
                self.model.train()
            elif early_stopping:
                self.history["val_loss"].append(None)

            print(log)

    def predict(self, df):
        for col in self.cat_columns:
            df[col] = self.encoders[col].transform(df[col].astype(str))
        df[self.num_columns] = self.scaler.transform(df[self.num_columns])
        
        cat_inputs = torch.tensor(df[self.cat_columns].values, dtype=torch.long).to(self.device)
        num_inputs = torch.tensor(df[self.num_columns].values, dtype=torch.float32).to(self.device)
        
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(cat_inputs, num_inputs)
        
        return predictions.cpu().numpy()


class ConvClassifierModel(ConvRegressorModel):
    def __init__(self, cat_dims, num_features, embedding_dim=4, conv_filters=[16, 32], dense_units=64):
        super(ConvClassifierModel, self).__init__(
            cat_dims, num_features, embedding_dim, conv_filters, dense_units
        )
        self.output_layer = nn.Linear(dense_units // 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, cat_inputs, num_inputs):
        x = super(ConvClassifierModel, self).forward(cat_inputs, num_inputs)
        return self.sigmoid(x)


class ConvClassifier(ConvRegressor):
    def build_model(self, df, target_col):
        self._identify_columns(df, target_col)
        df = self._preprocess(df)
        self.preprocessed_df = df

        cat_dims = [df[col].nunique() for col in self.cat_columns]
        num_features = len(self.num_columns)
        
        self.model = ConvClassifierModel(
            cat_dims=cat_dims,
            num_features=num_features,
            embedding_dim=self.embedding_dim,
            conv_filters=self.conv_filters,
            dense_units=self.dense_units
        ).to(self.device)
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters())
        
        return self.model


def train_test_val_split(df, target_cols, test_size=0.2, val_frac=0.1):
    df_trainval, df_test = train_test_split(df, test_size=test_size, random_state=42)
    df_train, df_val = train_test_split(df_trainval, test_size=val_frac, random_state=42)
    
    X_train = df_train.drop(columns=target_cols)
    X_val = df_val.drop(columns=target_cols)
    X_test = df_test.drop(columns=target_cols)
    
    y_train = df_train[target_cols]
    y_val = df_val[target_cols]
    y_test = df_test[target_cols]
    
    return X_train, y_train, X_val, y_val, X_test, y_test



class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        self.best_model_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        self.val_loss_min = val_loss


