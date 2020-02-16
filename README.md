## Forecasting Stock Price Movement using SEC 8K Fillings


### How it works?


#### Collecting Data
* Scrape 8-K documents from SEC Edgar database given a set of stock tickers
* Download historical open and close price data for the same stock and in addition GSPC index from Yahoo Finance.
* Calculate the stock return, based on when the corresponding doc was released (prior, during, post market hours), and subtract index return during the same time window


#### NLP & Deep Learning

* Apply the standard NLP techniques to process raw text data: tokenization, removing stop words, punctuation and numbers, lemmatization, and applying Glove 6B embedding vectors
* Training deep neural networks under two basic architectures: i) LSTM + global max pooling + dense layer and 2) vanilla Bidirectional LSTM


### How's it performs and how to improve?
Both two models provide minor improvement over random guess. While this can be simply due to the noisy nature of financial data, it suggests further refinements such as:
* Using richer data including all SP500 companies and training the model online (with GPU)!
* Replacing the fixed pre-trained word embeddings with the state-of-art model BERT
* Making more sensible response variable, e.g., using regression residuals from CAMP model (i.e., beta-adjusted normalized price return as opposed to the vanilla normalized return used in the project)
* Adding features from market data in addition to the text data
* More fine parameter tuning of the neural networks
