import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

plt.style.use("ggplot")


class RobertaPipeline:

    LABEL_MAPPING = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive",
    }

    def __init__(
        self,
        df: pl.DataFrame,
        model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
    ):
        """
        Initialize the pipeline with a DataFrame and a model name.

        Args:
            df (pl.DataFrame): Polars Dataframe with one of the columns named "Tweet".
            model_name (str, optional): Defaults to
            "cardiffnlp/twitter-roberta-base-sentiment".
        """
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, output_hidden_states=True
        )
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", model=self.model, tokenizer=self.tokenizer
        )

    def score_tweets(self, df: pl.DataFrame) -> list:
        """
        Score the tweets in the DataFrame.

        Args:
            df (pl.DataFrame): input Polars DataFrame with
            a column named "Tweet".

        Returns:
            list: list of tuples with the tweet and the sentiment score/label.
        """
        self.tweets = df["Tweet"].to_list()
        results = [
            (tweet, self.sentiment_pipeline(tweet))
            for tweet in tqdm(self.tweets, desc="Processing tweets")
        ]
        return results

    def get_embeddings(self) -> list:
        """
        Generate embeddings for the tweets.

        Returns:
            list: list of tensors with the embeddings
            for each tweet.
        """
        embeddings = []
        for tweet in tqdm(self.tweets, desc="Generating embeddings"):
            # Encode the tweets
            encoded_input = self.tokenizer(
                tweet,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**encoded_input)
            # Retrieve the full hidden states for the last layer
            tweet_embeddings = outputs.hidden_states[-1]
            embeddings.append(tweet_embeddings)
        return embeddings

    def get_cls_token(self, embeddings: list) -> np.ndarray:
        """
        Extract the CLS token for each tensor in the list of embeddings.

        Args:
            embeddings (list): list of tensors

        Returns:
            np.ndarray: array with the CLS token embeddings only
        """
        first_token_embeddings = []

        for tensor in tqdm(embeddings, desc="Extracting CLS token"):
            first_token_embedding = tensor[0, 0, :]
            first_token_embeddings.append(first_token_embedding.numpy())
        first_token_embeddings_matrix = np.vstack(first_token_embeddings)

        return first_token_embeddings_matrix

    def cluster_by_cls_token(
        self, first_token_embeddings_matrix: np.ndarray, n_clusters: int = 2
    ) -> list:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)

        kmeans.fit(first_token_embeddings_matrix)

        clusters = kmeans.labels_

        return clusters.tolist()

    def append_columns(
        self, df: pl.DataFrame, results: list, clusters: list
    ) -> pl.DataFrame:
        """
        Append the sentiment label and score to the DataFrame.

        Args:
            df (pl.DataFrame): original DataFrame
            results (list): list of tuples with the tweet and the
            sentiment score/label calculated by the model.

        Returns:
            pl.DataFrame: output DataFrame with the columns added
        """
        raw_labels = [item[1][0]["label"] for item in results]
        raw_scores = [item[1][0]["score"] for item in results]
        df = (
            df.with_columns(
                pl.Series(
                    name="Sentiment_Label",
                    values=[self.LABEL_MAPPING.get(label) for label in raw_labels],
                ),
            )
            .with_columns(pl.Series(name="Sentiment_Score", values=raw_scores))
            .with_columns(pl.Series(name="Cluster", values=clusters))
            .with_columns(
                pl.col("Cluster")
                .replace([1, 0], ["AAPL", "MSFT"])
                .alias("Cluster_mapped")
            )
        )
        return df

    def generate_confusion_matrix(self, df: pl.DataFrame) -> None:
        """
        Plot a confusion matrix to show the clustering result.

        Args:
            df (pl.DataFrame): input dataframe
        """
        labels = ["AAPL", "MSFT"]
        cm = confusion_matrix(
            df["Stock Name"], df["Cluster_mapped"], labels=labels, normalize="true"
        )

        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt=".2%", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.savefig("./output/confusion_matrix.png")

    def run(self) -> pl.DataFrame:
        """
        Run the pipeline.

        Returns:
            pl.DataFrame: output DataFrame with the columns added
        """
        df = self.df.clone()
        results = self.score_tweets(df)
        embeddings = self.get_embeddings()
        first_token_embeddings_matrix = self.get_cls_token(embeddings)
        clusters = self.cluster_by_cls_token(first_token_embeddings_matrix)
        df = self.append_columns(df, results, clusters)
        self.generate_confusion_matrix(df)
        print(df)
        return df


# Example usage

# if __name__ == "__main__":
#     df = pl.read_csv("./data/raw/stock_tweets.csv")
#     rb_pipeline = RobertaPipeline(df)
#     df_processed = rb_pipeline.run()
#     df_processed.write_csv("./data/intermediate/stock_tweets_with_sentiment.csv")
