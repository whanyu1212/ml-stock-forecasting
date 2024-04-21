from typing import Tuple

import numpy as np
import polars as pl
import tensorflow
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model


class LSTMClassifierPipeline:

    VARIABLES = [
        "Date",
        "Backward_Volatility",
        "Sentiment_lag_1",
        "Sentiment_lag_2",
        "Sentiment_lag_3",
        "Sentiment_lag_4",
        "Sentiment_lag_5",
        "Response_lag_1",
        "Response_lag_2",
        "Response_lag_3",
        "Response_lag_4",
        "Response_lag_5",
        "Sum_of_lagged_response",
        "Response",
    ]

    def __init__(self, df: pl.DataFrame):
        """
        Initialize the class function with the dataframe and the
        MinMaxScaler.

        Args:
            df (pl.DataFrame): input dataframe
        """
        self.df = df.drop_nulls().with_columns(
            pl.col("Date").str.to_datetime().dt.date()
        )
        self.scaler = MinMaxScaler()

    def train_validation_split(self, df: pl.DataFrame):
        train = df.filter(pl.col("Date") < pl.date(2022, 8, 1))
        val = df.filter(pl.col("Date") >= pl.date(2022, 8, 1))
        return train, val

    def preprocess_data(
        self, df: pl.DataFrame, stock_name1: str, stock_name2: str
    ) -> pl.DataFrame:
        """Do a concatenation of the two stock dataframes
        on Date and only keep the necessary columns
        Args:
            df (pl.DataFrame): input dataframe
            stock_name1 (str): stock name 1
            stock_name2 (str): stock name 2

        Returns:
            pl.DataFrame: preprocessed dataframe
        """
        return (
            df.filter(pl.col("Stock Name") == stock_name1)
            .select(self.VARIABLES)
            .join(
                df.filter(pl.col("Stock Name") == stock_name2).select(self.VARIABLES),
                on="Date",
            )
            .sort("Date")
            .drop("Date")
        )

    def preprocess_train_val(
        self, train: pl.DataFrame, val: pl.DataFrame
    ) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Apply the preprocessing steps to the train and val dataframes.

        Args:
            train (pl.DataFrame): training dataframe
            val (pl.DataFrame): validation dataframe

        Returns:
            Tuple[pl.DataFrame, pl.DataFrame]: a tuple of preprocessed train and
            val dataframes
        """
        train = self.preprocess_data(train, "AAPL", "MSFT")
        val = self.preprocess_data(val, "AAPL", "MSFT")
        return train, val

    def generate_feature_response(
        self, df: pl.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the data into features and response variables and scale
        the features using the MinMaxScaler.

        Args:
            df (pl.DataFrame): input dataframe

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: _description_
        """
        X_scaled = self.scaler.fit_transform(
            df.drop(["Response", "Response_right"]).to_numpy()
        )
        # X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        y1 = df["Response"].to_numpy()
        y2 = df["Response_right"].to_numpy()
        return X_scaled, y1, y2

    def create_sequences(
        self,
        features: np.ndarray,
        target1: np.ndarray,
        target2: np.ndarray,
        window_size: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences from given feature data and corresponding
        targets.

        Args:
        features (np.ndarray): Scaled feature data in numpy array format.
        target1 (np.ndarray): The first response variable as a numpy array.
        target2 (np.ndarray): The second response variable as a numpy array.
        window_size (int): The size of the window to use to create sequences.
        Default to 5.

        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple of numpy arrays
        (sequences, target1_sequences, target2_sequences).
        """
        sequences = []
        target1_sequences = []
        target2_sequences = []

        for i in range(len(features) - window_size + 1):
            sequences.append(features[i : i + window_size])

            target1_sequences.append(target1[i + window_size - 1])
            target2_sequences.append(target2[i + window_size - 1])

        return (
            np.array(sequences),
            np.array(target1_sequences),
            np.array(target2_sequences),
        )

    def create_model(
        self,
        window_size: int = 5,
        lstm_units: int = 100,
        dense_units: int = 128,
        optimizer: str = "adam",
    ) -> tensorflow.keras.Model:
        """
        Create a LSTM model for binary classification and compile it.

        Args:
            window_size (int, optional): Defaults to 5.
            lstm_units (int, optional): Defaults to 100.
            dense_units (int, optional): Defaults to 128.
            optimizer (str, optional): Defaults to "adam".

        Returns:
            tensorflow.keras.Model: LSTM model
        """
        input_layer = Input(shape=(window_size, 24))
        lstm_layer = LSTM(lstm_units)(input_layer)
        common_dense = Dense(dense_units, activation="relu")(lstm_layer)

        # Two separate output layers for binary classification
        output1 = Dense(1, activation="sigmoid", name="output1")(common_dense)
        output2 = Dense(1, activation="sigmoid", name="output2")(common_dense)

        model = Model(inputs=input_layer, outputs=[output1, output2])

        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss={"output1": "binary_crossentropy", "output2": "binary_crossentropy"},
            metrics={"output1": ["accuracy"], "output2": ["accuracy"]},
        )

        print(model.summary())

        return model

    def fit_model(
        self,
        model: tensorflow.keras.Model,
        X_sequences: np.ndarray,
        y1_sequences: np.ndarray,
        y2_sequences: np.ndarray,
    ) -> None:
        """
        Fit the model on the given sequences.

        Args:
            model (tensorflow.keras.Model): compiled LSTM model
            X_sequences (np.ndarray): input sequences
            y1_sequences (np.ndarray): output sequences for the first output
            y2_sequences (np.ndarray): output sequences for the second output
        """
        _ = model.fit(
            X_sequences, [y1_sequences, y2_sequences], epochs=20, batch_size=32
        )

    def evaluate_model(
        self,
        model: tensorflow.keras.Model,
        X_sequences: np.ndarray,
        y1_sequences: np.ndarray,
        y2_sequences: np.ndarray,
    ):
        predictions = model.predict(X_sequences)
        accuracy1 = accuracy_score(
            y1_sequences, predictions[0] > 0.5
        )  # Assuming sigmoid output, threshold is 0.5
        accuracy2 = accuracy_score(y2_sequences, predictions[1] > 0.5)

        print("Accuracy for the first output (y1):", accuracy1)
        print("Accuracy for the second output (y2):", accuracy2)

        report1 = classification_report(y1_sequences, predictions[0] > 0.5)
        report2 = classification_report(y2_sequences, predictions[1] > 0.5)

        print("Classification Report for First Output (y1):\n", report1)
        print("Classification Report for Second Output (y2):\n", report2)

    def run(self):
        """Run the flow."""
        df = self.df.clone()
        train, val = self.train_validation_split(df)
        train, val = self.preprocess_train_val(train, val)
        X_train_scaled, y1_train, y2_train = self.generate_feature_response(train)
        X_val_scaled, y1_val, y2_val = self.generate_feature_response(val)
        X_train_sequences, y1_train_sequences, y2_train_sequences = (
            self.create_sequences(X_train_scaled, y1_train, y2_train)
        )
        X_val_scaled_sequences, y1_val_sequences, y2_val_sequences = (
            self.create_sequences(X_val_scaled, y1_val, y2_val)
        )
        model = self.create_model()
        self.fit_model(model, X_train_sequences, y1_train_sequences, y2_train_sequences)
        self.evaluate_model(
            model, X_val_scaled_sequences, y1_val_sequences, y2_val_sequences
        )


if __name__ == "__main__":
    df = pl.read_csv("./data/processed/stock_combined.csv")
    pipeline = LSTMClassifierPipeline(df)
    pipeline.run()
