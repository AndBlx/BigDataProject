from os import path
from re import split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pattern.nl import sentiment
from PIL import Image
from sklearn import metrics
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import STOPWORDS, ImageColorGenerator, WordCloud

import clean_tweets
import mysql_class


# Getting all stored Tweets
tweets = mysql_class.get_all()
tweets = pd.DataFrame(tweets)

#Labelling with pattern.nl and converting into positive, negative and neutral
tweets['sentiment_pattern'] = tweets["content"].apply(lambda x: sentiment(x))
tweets['sentiment_pattern'] = tweets["sentiment_pattern"].apply(
    lambda x: str(x))
tweets['sentiment_pattern'] = tweets["sentiment_pattern"].apply(
    lambda x: x.split())
tweets['sentiment_pattern'] = tweets["sentiment_pattern"].apply(
    lambda x: x.pop(1))
tweets['sentiment_pattern'] = tweets['sentiment_pattern'].str.replace(')', '')
tweets['sentiment_pattern'] = tweets["sentiment_pattern"].apply(
    lambda x: float(x))


def labeling(sentiment):
    if sentiment > 0.4:
        sentiment = 'positive'
    elif sentiment < -0.4:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    return sentiment


tweets['sentiment_pattern_str'] = tweets.apply(
    lambda x: labeling(x['sentiment_pattern']), axis=1)

#Finished Dataframe
tweets


# Vader isn't working due to the limitation of the translation of the used Google API

# #Labelling with Vader
# analyzer = SentimentIntensityAnalyzer()

# analyzer.polarity_scores("VADER is slim, knap, en grappig.")
# analyzer.polarity_scores("VADER is slecht, ik hou er niet van. Je kunt het niet gebruiken.")


#----------------------------------------------------------------------#
# WordCloud

text = tweets['content'].values


wordcloud = WordCloud(width=1600, height=800,
                      background_color="white").generate(str(text))

# Display the generated image:
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# Frequency List
freq_list = tweets.content.str.split(
    expand=True).stack().value_counts().reset_index()
freq_list.columns = ['Word', 'Frequency']
freq_list.iloc[0:50, :]


#--------------------------------------------------#

# Testing pattern.nl with Kaggle Data


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

dutch_tweets

dutch_tweets['sentiment_pattern'] = dutch_tweets["full_text"].apply(lambda x: sentiment(x))

dutch_tweets['sentiment_pattern'] = dutch_tweets["sentiment_pattern"].apply(
    lambda x: str(x))
dutch_tweets['sentiment_pattern'] = dutch_tweets["sentiment_pattern"].apply(
    lambda x: x.split())
dutch_tweets['sentiment_pattern'] = dutch_tweets["sentiment_pattern"].apply(
    lambda x: x.pop(1))
dutch_tweets['sentiment_pattern'] = dutch_tweets['sentiment_pattern'].str.replace(')', '')
dutch_tweets['sentiment_pattern'] = dutch_tweets["sentiment_pattern"].apply(
    lambda x: float(x))


def labeling(sentiment):
    if sentiment > 0.4:
        sentiment = 'positive'
    elif sentiment < -0.4:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    return sentiment


dutch_tweets['sentiment_pattern'] = dutch_tweets.apply(
    lambda x: labeling(x['sentiment_pattern']), axis=1)


# Results
print("Accuracy: ", round(
    metrics.accuracy_score(dutch_tweets['sentiment'], dutch_tweets['sentiment_pattern']), 3))
print("F1: ", round(metrics.f1_score(dutch_tweets['sentiment'], dutch_tweets['sentiment_pattern'], average='macro'), 3))
print("Precision: ", round(
    metrics.precision_score(dutch_tweets['sentiment'], dutch_tweets['sentiment_pattern'], average='macro'), 3))
print("Recall: ", round(metrics.recall_score(dutch_tweets['sentiment'], dutch_tweets['sentiment_pattern'], average='macro'), 3))

# Matrix
print("Confusion Matrix:")
cnf_matrix = metrics.confusion_matrix(dutch_tweets['sentiment'], dutch_tweets['sentiment_pattern'], labels=["positive", "neutral", "negative"])
cnf_matrix

dutch_tweets.groupby(['sentiment', "sentiment_pattern"]).size()