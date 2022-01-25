from nltk.corpus import stopwords

def ascii_converter(data):
    data['text'] = data['text'].str.encode(
        "ascii", "ignore").str.decode("ascii")
    data['user'] = data['user'].str.encode(
        "ascii", "ignore").str.decode("ascii")
    return data


def normalization(data):
    data['text'] = data["text"].str.lower()
    return data


def process_tweet(data):
    # Remove old style retweet text "RT"
    data['text'] = data['text'].str.replace(r'^RT[\s]', '')

    # Remove hyperlinks
    data['text'] = data['text'].str.replace(r'https?:\/\/.*[\r\n]*', '')

    # Remove punctuation
    data['text'] = data['text'].str.replace(r'[^\w\s]+', '')

    # Remove username
    data['text'] = data['text'].str.replace(r'@[\w]+', '')

    # Import the dutch stop words list from NLTK
    stopwords_dutch = stopwords.words('dutch')

    data['text'] = data["text"].apply(lambda x: ' '.join(
        [item for item in x.split() if item not in stopwords_dutch]))

    return data
