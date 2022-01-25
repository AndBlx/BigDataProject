def get_tweets(search_word):
    import tweepy
    import pandas as pd
    consumer_key = '-'
    consumer_secret = '-'
    access_token = '-'
    access_token_secret = '-'
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    date_since = "2021-9-10"

    api = tweepy.API(auth)

    tweets = tweepy.Cursor(api.search,
                           q=search_word + "-filter:retweets",
                           lang="nl", tweet_mode='extended',
                           since=date_since).items(2000)
    tweets

    users_locs = [[tweet.user.id, tweet.user.name, tweet.created_at,
                   tweet.id_str, tweet.full_text] for tweet in tweets]
    users_locs

    tweet_text = pd.DataFrame(data=users_locs,
                              columns=['user_id', 'user', 'timestamp', 'id_str', 'text'])
    return tweet_text
