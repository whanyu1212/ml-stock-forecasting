import polars as pl


class DataMorphosis:
    def __init__(self, df_stock, df_tweets):
        self.df_stock = df_stock.drop_nulls().with_columns(
            pl.col("Date").str.to_date("%m/%d/%Y")
        )
        self.df_tweets = df_tweets.drop_nulls().with_columns(
            pl.col("Date").str.to_datetime().dt.date()
        )

    def transform_sentiment_scores(self, df_tweets: pl.DataFrame) -> pl.DataFrame:
        df_tweets = df_tweets.with_columns(
            Sentiment_Score_transformed=pl.when(pl.col("Sentiment_Label") == "positive")
            .then(pl.col("Sentiment_Score") * 1)
            .when(pl.col("Sentiment_Label") == "negative")
            .then(pl.col("Sentiment_Score") * -1)
            .otherwise(0)
        )
        return df_tweets

    def agg_daily_sentiment_score(self, df_tweets: pl.DataFrame) -> pl.DataFrame:
        df_tweets_agg = (
            df_tweets.group_by(["Date", "Stock Name"])
            .agg(pl.median("Sentiment_Score_transformed"))
            .sort("Date")
        )

        print(df_tweets_agg.head(10))

        return df_tweets_agg

    def join_price_with_sentiment(
        self, df_stock: pl.DataFrame, df_tweets_agg: pl.DataFrame
    ) -> pl.DataFrame:
        df_combined = df_stock.join(
            df_tweets_agg, on=["Date", "Stock Name"], how="left"
        ).fill_nan(0)

        return df_combined

    def calculate_daily_returns(self, df_sub: pl.DataFrame) -> pl.DataFrame:
        df_sub = df_sub.with_columns(
            Return=(pl.col("Close") / pl.col("Close").shift(1)) - 1
        )

        return df_sub

    def calculate_backward_volatility(
        self, df_sub: pl.DataFrame, window: int = 5
    ) -> pl.DataFrame:
        df_sub = df_sub.with_columns(
            pl.col("Return")
            .rolling_std(window_size=window)
            .shift(1)
            .alias("Backward_Volatility")
        )

        return df_sub

    def calculate_forward_volatility(
        self, df_sub: pl.DataFrame, window: int = 5
    ) -> pl.DataFrame:
        df_sub = df_sub.with_columns(
            pl.col("Return")
            .reverse()
            .rolling_std(window)
            .reverse()
            .alias("Forward_Volatility")
        )

        return df_sub

    def create_lagged_sentiment(
        self, df_sub: pl.DataFrame, window: int = 5
    ) -> pl.DataFrame:
        for i in range(1, window + 1):
            df_sub = df_sub.with_columns(
                pl.col("Sentiment_Score_transformed").alias(f"Sentiment_lag_{i}")
            )
        return df_sub

    def create_lagged_response(
        self, df_combined_new: pl.DataFrame, window: int = 5
    ) -> pl.DataFrame:
        for i in range(1, window + 1):
            df_combined_new = df_combined_new.with_columns(
                pl.col("Response").shift(i).alias(f"Response_lag_{i}")
            )
        df_combined_new = df_combined_new.with_columns(
            Sum_of_lagged_response=pl.col("Response_lag_1")
            + pl.col("Response_lag_2")
            + pl.col("Response_lag_3")
            + pl.col("Response_lag_4")
            + pl.col("Response_lag_5")
        )
        return df_combined_new

    def create_additional_features(self, df_combined: pl.DataFrame) -> pl.DataFrame:
        df_aapl = df_combined.filter(pl.col("Stock Name") == "AAPL")
        df_msft = df_combined.filter(pl.col("Stock Name") == "MSFT")

        df_aapl = self.calculate_daily_returns(df_aapl)
        df_aapl = self.calculate_backward_volatility(df_aapl)
        df_aapl = self.calculate_forward_volatility(df_aapl)
        df_aapl = self.create_lagged_sentiment(df_aapl)

        df_msft = self.calculate_daily_returns(df_msft)
        df_msft = self.calculate_backward_volatility(df_msft)
        df_msft = self.calculate_forward_volatility(df_msft)
        df_msft = self.create_lagged_sentiment(df_msft)

        df_combined_new = df_aapl.vstack(df_msft)

        max_returns = df_combined_new.groupby("Date").agg(
            pl.max("Return").alias("MaxReturn")
        )

        df_combined_new = df_combined_new.join(max_returns, on="Date")

        df_combined_new = df_combined_new.with_columns(
            Response=(pl.col("Return") == pl.col("MaxReturn")).cast(pl.UInt8)
        ).drop("MaxReturn")

        df_combined_new = self.create_lagged_response(df_combined_new)

        return df_combined_new

    def run(self):
        df_tweets = self.df_tweets.clone()
        df_stock = self.df_stock.clone()
        df_tweets = self.transform_sentiment_scores(self.df_tweets)
        df_tweets_agg = self.agg_daily_sentiment_score(df_tweets)
        df_combined = self.join_price_with_sentiment(df_stock, df_tweets_agg)
        df_combined_new = self.create_additional_features(df_combined)
        print(df_combined_new.head(10))
        return df_combined_new


if __name__ == "__main__":
    df_stock = pl.read_csv("./data/raw/stock_yfinance_data.csv")
    df_tweets = pl.read_csv("./data/intermediate/stock_tweets_with_sentiment.csv")
    processor = DataMorphosis(df_stock, df_tweets)
    df_combined = processor.run()
    df_combined.write_csv("./data/processed/stock_combined.csv")
