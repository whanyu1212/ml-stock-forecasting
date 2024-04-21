import polars as pl
from loguru import logger

from src.data_morphosis import DataMorphosis
from src.lgb_classification import LGBMClassifierPipeline
from src.lgb_regressor import LGBMRegressorPipeline
from src.lstm_classification import LSTMClassifierPipeline
from src.processing_roberta import RobertaPipeline


def load_data(path: str) -> pl.DataFrame:
    """
    Load data from a CSV file into a Polars DataFrame.

    Args:
        path (str): path to the CSV file.

    Returns:
        pl.DataFrame: Polars DataFrame.
    """
    return pl.read_csv(path)


def run_roberta_pipeline(df: pl.DataFrame):
    """
    Run the RoBERTa pipeline.

    Args:
        df (pl.DataFrame): input Polars DataFrame.
    """
    rb_pipeline = RobertaPipeline(df)
    df_processed = rb_pipeline.run()
    df_processed.write_csv("./data/intermediate/stock_tweets_with_sentiment.csv")
    return df_processed


def run_data_morphosis_pipeline(df_stock, df_tweets):
    processor = DataMorphosis(df_stock, df_tweets)
    df_combined = processor.run()
    df_combined.write_csv("./data/processed/stock_combined.csv")
    return df_combined


def run_lgb_classification_pipeline(df: pl.DataFrame):
    """
    Run the LGBM classification pipeline.

    Args:
        df (pl.DataFrame): input Polars DataFrame.
    """
    pipeline = LGBMClassifierPipeline(df)
    pipeline.run()


def run_lstm_classification_pipeline(df: pl.DataFrame):
    """
    Run the LSTM classification pipeline.

    Args:
        df (pl.DataFrame): input Polars DataFrame.
    """
    pipeline = LSTMClassifierPipeline(df)
    pipeline.run()


def run_lgb_regressor_pipeline(df: pl.DataFrame):
    """
    Run the LGBM regression pipeline.

    Args:
        df (pl.DataFrame): input Polars DataFrame.
    """
    pipeline = LGBMRegressorPipeline(df)
    pipeline.run()


def main():
    logger.info("Loading data...")
    df_tweets_raw = load_data("./data/raw/stock_tweets.csv")
    df_stock_raw = load_data("./data/raw/stock_yfinance_data.csv")
    logger.info("Processing data...")
    df_tweets_processed = run_roberta_pipeline(df_tweets_raw)
    df_combined = run_data_morphosis_pipeline(df_stock_raw, df_tweets_processed)
    logger.success("data processing done!")
    logger.info("Running pipelines...")
    run_lgb_classification_pipeline(df_combined)
    run_lstm_classification_pipeline(df_combined)
    run_lgb_regressor_pipeline(df_combined)
    logger.success("All pipelines done!")


if __name__ == "__main__":
    main()
