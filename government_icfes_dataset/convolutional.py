import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Input, MaxPooling1D, Embedding, Dense, Concatenate, Conv1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


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
    
    def _identify_columns(self, df, target_col):
        self.cat_columns = df.select_dtypes(include=['object', 'category']).columns.drop(target_col, errors='ignore').tolist()
        self.num_columns = df.select_dtypes(include=[np.number]).columns.drop(target_col, errors='ignore').tolist()

    def _preprocess(self, df):
        # Label Encode categóricas
        df = df.dropna(ignore_index=True)

        for col in self.cat_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le

        # Escalar numéricas
        self.scaler = StandardScaler()
        df[self.num_columns] = self.scaler.fit_transform(df[self.num_columns])
        return df

    def build_model(self, df, target_col):
        self._identify_columns(df, target_col)
        df = self._preprocess(df)
        self.preprocessed_df = df  


        inputs = []
        embeddings = []

        # Categóricas → Embeddings
        for col in self.cat_columns:
            input_cat = Input(shape=(1,), name=f"input_{col}")
            vocab_size = df[col].nunique()
            emb = Embedding(input_dim=vocab_size + 1, output_dim=self.embedding_dim, name=f"emb_{col}")(input_cat)
            emb = Flatten()(emb)
            inputs.append(input_cat)
            embeddings.append(emb)

        # Numéricas
        input_num = Input(shape=(len(self.num_columns),), name="input_numeric")
        inputs.append(input_num)
        embeddings.append(input_num)

        x = Concatenate()(embeddings)
        x = tf.expand_dims(x, axis=-1) 

        for f in self.conv_filters:
            x = Conv1D(filters=f, kernel_size=3, padding="same")(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(0.2)(x)

        x = Conv1D(filters=64, kernel_size=3, padding="same")(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = Dropout(0.3)(x)

        x = Flatten()(x)
        x = Dense(self.dense_units, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(self.dense_units // 2, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(1, name="regression_output")(x)

        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return self.model


    def train(self, df=None, target_col=None, epochs=30, batch_size=16):
        if self.model is None:
            raise ValueError("El modelo no ha sido construido. Llama a build_model() primero.")

        if df is None:
            if self.preprocessed_df.empty:
                raise ValueError("No hay DataFrame preprocesado disponible. Llama primero a build_model() con un DataFrame.")
            df = self.preprocessed_df

        X_inputs = [df[col].values for col in self.cat_columns] + [df[self.num_columns].values]
        y = df[target_col].values

        self.model.fit(X_inputs, y, epochs=epochs, batch_size=batch_size, verbose=1)


    def predict(self, df):
        for col in self.cat_columns:
            df[col] = self.encoders[col].transform(df[col].astype(str))
        df[self.num_columns] = self.scaler.transform(df[self.num_columns])

        X_inputs = [df[col].values for col in self.cat_columns] + [df[self.num_columns].values]
        return self.model.predict(X_inputs)


class ConvClassifier(ConvRegressor):
    def build_model(self, df, target_col):
        self._identify_columns(df, target_col)
        df = self._preprocess(df)
        self.preprocessed_df = df

        inputs = []
        embeddings = []

        for col in self.cat_columns:
            input_cat = Input(shape=(1,), name=f"input_{col}")
            vocab_size = df[col].nunique()
            emb = Embedding(input_dim=vocab_size + 1, output_dim=self.embedding_dim, name=f"emb_{col}")(input_cat)
            emb = Flatten()(emb)
            inputs.append(input_cat)
            embeddings.append(emb)

        input_num = Input(shape=(len(self.num_columns),), name="input_numeric")
        inputs.append(input_num)
        embeddings.append(input_num)

        x = Concatenate()(embeddings)
        x = tf.expand_dims(x, axis=-1)

        for f in self.conv_filters:
            x = Conv1D(filters=f, kernel_size=3, padding="same")(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(0.2)(x)

        x = Conv1D(filters=64, kernel_size=3, padding="same")(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = Dropout(0.3)(x)

        x = Flatten()(x)
        x = Dense(self.dense_units, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(self.dense_units // 2, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(1, activation="sigmoid", name="classification_output")(x)

        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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



def run_kfold_cv(model_class, df, target_col, task_type='regression', k=5, epochs=40):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\nFold {fold+1}/{k}")

        df_train = df.iloc[train_idx].drop(columns=['punt_matematicas', 'eco'])
        df_val = df.iloc[val_idx].drop(columns=['punt_matematicas', 'eco'])

        df_train[target_col] = df.iloc[train_idx][target_col]
        df_val[target_col] = df.iloc[val_idx][target_col]

        model = model_class()
        model.build_model(df_train.copy(), target_col)
        model.train(target_col=target_col, epochs=epochs, batch_size=32)

        y_true = df_val[target_col].values
        y_pred = model.predict(df_val.copy()).squeeze()

        if task_type == 'regression':
            score = tf.keras.metrics.mean_squared_error(y_true, y_pred).numpy().mean()
            print(f"MSE Fold {fold+1}: {score:.2f}")
        else:
            score = tf.keras.metrics.binary_accuracy(y_true, y_pred.round()).numpy().mean()
            print(f"Accuracy Fold {fold+1}: {score:.4f}")

        fold_results.append(score)
    
    print(f"\nPromedio final: {np.mean(fold_results):.4f}")



def train_final_model(model_class, df_train, target_col, epochs=80, patience=10):
    model = model_class()
    model.build_model(df_train.copy(), target_col)
    
    callback = EarlyStopping(monitor='loss', patience=patience, restore_best_weights=True)
    model.train(target_col=target_col, epochs=epochs, batch_size=32)
    return model
