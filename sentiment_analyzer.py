import pickle
import time
from os import path
from re import split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pattern.nl import sentiment
from PIL import Image
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import STOPWORDS, ImageColorGenerator, WordCloud

import clean_tweets
import mysql_class

#Getting all the Tweets from json and concat them
dutch_tweets = pd.DataFrame()
for x in range(9):
    x = str(x)
    file = pd.read_json(str('archive_dutch/dutch_tweets_chunk'+x+'.json'))
    dutch_tweets = pd.concat([file, dutch_tweets])
    pd.DataFrame(dutch_tweets)
    df = dutch_tweets.reset_index(drop=True)

#Dropping unused cols
dutch_tweets = dutch_tweets[['full_text', 'sentiment_pattern']]
dutch_tweets

#Converting sentiment from float into positive, negative and neutral
def labeling(sentiment):
    if sentiment > 0.2:
        sentiment = 1
    elif sentiment < -0.05:
        sentiment = -1
    else:
        sentiment = 0
    return sentiment


dutch_tweets['sentiment'] = dutch_tweets.apply(
    lambda x: labeling(x['sentiment_pattern']), axis=1)

#Rename Dataframe
df = dutch_tweets

#Renaming DataFrames column
df = df.rename({'full_text': 'text'}, axis=1)

#Cleaning and preprocessing data
df['text'] = df['text'].str.encode("ascii", "ignore").str.decode("ascii")
df = clean_tweets.normalization(df)
df = clean_tweets.process_tweet(df)
df = df[df['text'].notna()]

# Train & Test split
train, test = train_test_split(df, test_size=0.2, random_state=42)

# CountVectorizer with params
vecCV = CountVectorizer(binary=True, analyzer='word', ngram_range=(1, 2))

#Vectorizing train and test set
train_vectors_CV = vecCV.fit_transform(train['text'].values)
test_vectors_CV = vecCV.transform(test['text'].values)

#Building and fitting of NB
model_NB = MultinomialNB(alpha=1)
model_NB.fit(train_vectors_CV, train['sentiment'])
y_pred = model_NB.predict(test_vectors_CV)

# Results
print("Accuracy: ", round(
    metrics.accuracy_score(test['sentiment'], y_pred), 3))

print("F1: ", round(metrics.f1_score(
    test['sentiment'], y_pred, average='macro'), 3))
print("Precision: ", round(
    metrics.precision_score(test['sentiment'], y_pred,  average='macro'), 3))
print("Recall: ", round(metrics.recall_score(
    test['sentiment'], y_pred,  average='macro'), 3))

# Matrix
print("Confusion Matrix:")
cnf_matrix = metrics.confusion_matrix(test['sentiment'], y_pred)
cnf_matrix


# Saving the model and vectorizer
pickle.dump(model_NB, open('model_NB.sav', 'wb'))
pickle.dump(vecCV, open('Vektorizer.sav', 'wb'))

