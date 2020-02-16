from keras.layers import Dense, Dropout, LSTM, Bidirectional, GlobalMaxPooling1D
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import pickle
import re
import string


stop_words = stopwords.words("english")
punctuations = string.punctuation
wnl = WordNetLemmatizer()
MAX_LEN = 10000
EMBEDDING_DIM = 100


def normalize_one_doc(text):
    """
    normalize texts in one doc
    """
    try:
        tokens = [word for word in word_tokenize(text) if word.isalpha()]
        tokens = list(filter(lambda t: t not in punctuations, tokens))
        tokens = list(filter(lambda t: t.lower() not in stop_words, tokens))
        filtered_tokens = []
        for token in tokens:
            if re.search("[a-zA-Z]", token):
                filtered_tokens.append(token)
        filtered_tokens = list(map(lambda token: wnl.lemmatize(token.lower()), filtered_tokens))
        filtered_tokens = list(filter(lambda t: t not in punctuations, filtered_tokens))
        return filtered_tokens
    except Exception as e:
        raise e


def normalize_docs(df, save=True):
    """
    apply text normalization to all docs
    """
    df["cleaned_text"] = df["text"].map(normalize_one_doc)
    df["text_len"] = df["cleaned_text"].map(lambda x: len(x))
    
    if save:
        df.to_pickle("normalized_data.pkl")
        
    return df


def load_embeddings(file="glove.6B.100d.txt"):
    """
    load pre-train word embedding
    """
    f = open(file)
    embed_index = {}
    for line in f:
        val = line.split()
        word = val[0]
        coff = np.asarray(val[1:], dtype ="float")
        embed_index[word] = coff
    f.close()
    print('Found %s word vectors.' % len(embed_index))
    return embed_index



def word2vec(docs, embed_index, embed_dim=EMBEDDING_DIM, max_len=MAX_LEN):
    """
    apply Glove 6b embedding vectors to word sequences
    """
    X = np.zeros((len(docs), max_len, 100))
    for i, item in enumerate(docs):
        for j, word in enumerate(item):
            if j < max_len:
                temp = embed_index.get(word)
                if temp is not None :
                    X[i, j, :] = temp
    return X


def create_modeling_data(df):
    """
    wrapper function that prepares data ready for training neural network
    """
    # get response and features
    y = df["normalized_change"].apply(lambda x: 1 if x >= 0 else 0)    
    docs = df['cleaned_text']
    
    # train and test split
    y_train, y_test, docs_train, docs_test = train_test_split(y, docs, stratify=y, test_size=0.3, random_state=20)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    docs_train.reset_index(drop=True, inplace=True)
    docs_test.reset_index(drop=True, inplace=True)
    
    # handling imbalanced labels
    y_train, y_test, docs_train, docs_test = y_train.values, y_test.values, docs_train.values, docs_test.values
    
    # apply Glove word vectors
    embed_index = load_embeddings()        
    docs_train, docs_test = word2vec(docs_train, embed_index), word2vec(docs_test, embed_index)
    
    return y_train, y_test, docs_train, docs_test


def build_model(option="LSTM"):
    """
    build neural network
    """
    model = Sequential()
    if option == "LSTM":
        model.add(LSTM(32, return_sequences=True, input_shape=(MAX_LEN, EMBEDDING_DIM)))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.2))
    elif option == "BiLSTM":
        model.add(Bidirectional(LSTM(32, input_shape=(MAX_LEN, EMBEDDING_DIM))))    
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return model


def evaluate_model(model, X, y):
    """
    evaluate model given truth
    """
    y_pred = model.predict(X).flatten()
    acc = ((1 * (y_pred>=0.5)) == y).mean()
    auc = roc_auc_score(y, y_pred)
    print("Accuracy: {}; AUC: {}".format(acc, auc))


if __name__ == "__main__":
    df = pd.read_pickle("doc_and_financial_data.pkl")
    
    # Step 1. Clean up docs and save each doc as a list of lemmatized words
    df = normalize_docs(df)
    
    # Step 2. Prepare data for modeling
    y_train, y_test, docs_train, docs_test = create_modeling_data(df)
    
    # Step 3. Training neural network
    model_lstm = build_model("LSTM")
    model_lstm.fit(docs_train, y_train, epochs=10, batch_size=64)
    evaluate_model(model_lstm, docs_test, y_test)
    
    model_bilstm = build_model("BiLSTM")
    model_bilstm.fit(docs_train, y_train, epochs=10, batch_size=64)
    evaluate_model(model_bilstm, docs_test, y_test)
