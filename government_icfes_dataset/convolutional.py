import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.layers import Input, MaxPooling1D, Embedding, Dense, Concatenate, Conv1D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow as tf

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
    
    def _identify_columns(self, df, target_col):
        self.cat_columns = df.select_dtypes(include=['object', 'category']).columns.drop(target_col, errors='ignore').tolist()
        self.num_columns = df.select_dtypes(include=[np.number]).columns.drop(target_col, errors='ignore').tolist()

    def _preprocess(self, df):
        # Label Encode categóricas
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

        # Concatenación final
        x = Concatenate()(embeddings)
        x = tf.expand_dims(x, axis=-1)  # (batch_size, features, 1)

        # Bloques Conv1D + MaxPooling1D
        for f in self.conv_filters:
            x = Conv1D(filters=f, kernel_size=3, padding="same")(x)
            x = BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(0.2)(x)

        # Bloque final extra de convolución
        x = Conv1D(filters=64, kernel_size=3, padding="same")(x)
        x = BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = Dropout(0.3)(x)

        # Aplanar y pasar por más capas densas
        x = Flatten()(x)
        x = Dense(self.dense_units, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(self.dense_units // 2, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(1, name="regression_output")(x)

        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return self.model


    def train(self, df, target_col, epochs=30, batch_size=16):
        if self.model is None:
            self.build_model(df, target_col)
        
        X_inputs = [df[col].values for col in self.cat_columns] + [df[self.num_columns].values]
        y = df[target_col].values

        self.model.fit(X_inputs, y, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, df):
        for col in self.cat_columns:
            df[col] = self.encoders[col].transform(df[col].astype(str))
        df[self.num_columns] = self.scaler.transform(df[self.num_columns])

        X_inputs = [df[col].values for col in self.cat_columns] + [df[self.num_columns].values]
        return self.model.predict(X_inputs)

