## Sentiment Analysis and Volatility Forecasting

This project explores the link between Twitter-expressed stock sentiment and future market volatility. It primarily targets two outcomes: forecasting upcoming volatility and determining which of two stocks, AAPL or MSFT, will yield a higher return on a given day. For the return prediction, we employ both the LGBMClassifier and a multi-output LSTM model. The LGBMRegressor is used to predict forward-looking volatility.

<br/>

## Components:
1. `processing_roberta.py`: The `RobertaPipeline` classes process the tweets using the transformer pipeline and assign them with sentiments scores. Following that, we will calculate the embeddings (A list of tensors) and use the CLS token to perform the clustering. The confusion matrix is stored as `./output/confusion_matrx.png`
2. `data_morphosis.py`: Scales the sentiment scores according to their labels. The processed sentiment scores are aggregated at the daily level before joining with the stock pricing data. Returns, backward volatility, forward volatiltiy and a couple of lagged variables are also created and appended to the dataframe prior to modelling.
3. `lgbm_classification.py`: Constructs a LGBM classification model that uses `Backward_Volatility`, `Sentiment_lag_1`, `Sentiment_lag_2`, `Sentiment_lag_3`, `Sentiment_lag_4`, `Sentiment_lag_5`, `Response_lag_1`, `Response_lag_2`, `Response_lag_3`, `Response_lag_4`, `Response_lag_5`, `Sum_of_lagged_response` to predict `Response` (0,1). The accuracy is around **0.55**
4. `lstm_classification.py`: Constructs a multi-output LSTM model (using the same set of features), y_1 being the response for AAPL and y_2 being the response for MSFT. The accuracy is around **0.49** and **0.46** respectively
5. `lgb_regressor.py`: Constructs a regression model (using the same set of features) to predict forward volatility. R2 is  **-0.844** (time series split) which goes to show that the features have no predictive power.