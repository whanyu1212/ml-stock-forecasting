import polars as pl
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class LGBMRegressorPipeline:
    # class level constants
    PREDICTORS = [
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
    ]
    RESPONSE = "Forward_Volatility"

    def __init__(self, df: pl.DataFrame):
        self.df = df.drop_nulls()
        self.model = LGBMRegressor(
            n_estimators=1000, learning_rate=0.01, random_state=42
        )

    def train_validation_split(self, df: pl.DataFrame):
        train = df.filter(pl.col("Date") < "2022-08-01")
        val = df.filter(pl.col("Date") >= "2022-08-01")
        return train, val

    def select_predictors_response(self, train, val):
        X_train = train.select(self.PREDICTORS)
        y_train = train.select(self.RESPONSE)
        X_val = val.select(self.PREDICTORS)
        y_val = val.select(self.RESPONSE)

        return X_train, y_train, X_val, y_val

    def fit_model(self, X_train, y_train):
        self.model.fit(X_train.to_numpy(), y_train.to_numpy().flatten())

    def evaluate_model(self, X_val, y_val):
        y_pred = self.model.predict(X_val.to_numpy())
        r2 = r2_score(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        return r2, mse, mae

    def run(self):
        df = self.df.clone()
        train, val = self.train_validation_split(df)
        X_train, y_train, X_val, y_val = self.select_predictors_response(train, val)
        self.fit_model(X_train, y_train)
        r2, mse, mae = self.evaluate_model(X_val, y_val)
        print(f"R2: {r2}")
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")


if __name__ == "__main__":
    df = pl.read_csv(
        "/Users/hanyuwu/Study/stock-forecasting/data/intermediate/stock_combined.csv"
    )
    pipeline = LGBMRegressorPipeline(df)
    pipeline.run()
