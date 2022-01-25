import time
import pandas as pd
import clean_tweets
import mysql_class
import Twwepy
import pickle

from datetime import datetime

# Request Tweets from Twitter (5) -  "#covid", "#covid19",
searchList = ["#2g", "#2G", "#covid", "#covid-19", "#stayhomestaysafe", "#coronavirus", "#stayhome",
              "#quarantineandchill", "#lockdownnow", "#covidiots", "#socialdistancing, #vaccination", "#lockdown"]
for search_word in searchList:
    df = Twwepy.get_tweets(search_word)
    print("Query done")
    # Loading Tweets into DB
    df = clean_tweets.ascii_converter(df)
    df = clean_tweets.normalization(df)

    df['category'] = df["text"].map(
        lambda x: [item for item in x.split() if item in searchList])

    df = clean_tweets.process_tweet(df)
    df.apply(lambda x: mysql_class.add_post_db(str(x.id_str), str(
        x.user_id), x.user, 
        datetime.strptime(str(x.timestamp), '%Y-%m-%d %H:%M:%S'), 
        x.text, x.category), axis=1)

    print("Loading done")
    time.sleep(1500)
    print("Sleep done")

vectorizer = pickle.load(open('models/Vektorizer.sav', 'rb'))
model = pickle.load(open('models/model_NB.sav', 'rb'))
mysql_class.update_sentiment(model, vectorizer)

print("********** updater sucess ************")