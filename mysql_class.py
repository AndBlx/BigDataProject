from datetime import time
from numpy import mod
import pandas as pd

import sqlalchemy
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean, ForeignKey, select
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import backref, relationship, sessionmaker

username = "oudshob2"
password = "TXs3fpOIPdIt$q"
server = "oege.ie.hva.nl"
port = "3306"
database = "zoudshob2"

engine = create_engine(
    'mysql+mysqlconnector://' + username + ':' + password + '@' + server + ':' + port + '/' + database + '')
Session = sessionmaker(bind=engine, autoflush=False)

Base = declarative_base()


class Post(Base):
    __tablename__ = 'Post'

    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey('User.id'))
    category_id = Column(Integer, ForeignKey('Category.id'))
    timestamp = Column(DateTime)
    content = Column(String)
    sentiment = Column(Integer)

    user = relationship("User", backref=backref('Post'))
    category = relationship("Category", backref=backref('Post'))

    def __init__(self, id, user, category, timestamp, content, sentiment):
        self.id = id
        self.timestamp = timestamp
        self.content = content
        self.sentiment = sentiment
        self.user = user
        self.category = category


class User(Base):
    __tablename__ = 'User'

    id = Column(String, primary_key=True)
    name = Column(String)
    location_key = Column(Integer)

    def __init__(self, id,  name):
        self.id = id
        self.name = name


class Category(Base):
    __tablename__ = 'Category'

    id = Column(Integer, primary_key=True)
    covid = Column(Boolean)
    covid19 = Column(Boolean)
    covid_dash_19 = Column(Boolean)
    stayHomeStaySafe = Column(Boolean)
    coronavirus = Column(Boolean)
    stayHome = Column(Boolean)
    quarantineandChill = Column(Boolean)
    lockdownNow = Column(Boolean)
    covidiots = Column(Boolean)
    socialDistancing = Column(Boolean)
    vaccination = Column(Boolean)
    lockdown = Column(Boolean)

    def __init__(self, covid, covid19, covid_dash_19, stayHomeStaySafe, coronavirus, stayHome, quarantineandChill,
                 lockdownNow, covidiots, socialDistancing, vaccination, lockdown):
        self.covid = covid
        self.covid19 = covid19
        self.covid_dash_19 = covid_dash_19
        self.stayHomeStaySafe = stayHomeStaySafe
        self.coronavirus = coronavirus
        self.stayHome = stayHome
        self.quarantineandChill = quarantineandChill
        self.lockdownNow = lockdownNow
        self.covidiots = covidiots
        self.socialDistancing = socialDistancing
        self.vaccination = vaccination
        self.lockdown = lockdown


def get_category(hashtags):
    cat = {"covid": 0, "covid19": 0, "covid-19": 0, "stayhomestaysafe": 0, "coronavirus": 0, "stayhome": 0,
           "quarantineandchill": 0, "lockdownnow": 0, "covidiots": 0, "socialdistancing": 0, "vaccination": 0, "lockdown": 0}
    for hastag in hashtags:
        if hastag in cat:
            cat[hastag] = 1
    return Category(cat["covid"], cat["covid19"], cat["covid-19"], cat["stayhomestaysafe"], cat["coronavirus"], cat["stayhome"], cat["quarantineandchill"],
                    cat["lockdownnow"], cat["covidiots"], cat["socialdistancing"], cat["vaccination"], cat["lockdown"])


def add_post_db(id, user_id, username, timestamp, content, hastags):
    session = Session()
    try:
        user_object = User(user_id, username)
        user = get_or_create(session, User, user_id, user_object)
        category = get_category(hastags)
        post = Post(id, user, category, timestamp, content, None)
        session.add(category)
        session.add(post)
        session.commit()
    except sqlalchemy.exc.IntegrityError:
        session.rollback()
        print("post already exists")
    finally:
        session.close()

# checks if tweet is already exists when it does it returns the instance of the tweet
# otherwise it will create the tweet in the database
def get_or_create(session, model, search_id, object):
    instance = session.query(model).filter_by(id=search_id).first()
    if instance:
        return instance
    else:
        instance = object
        session.add(instance)
        return instance

import transformers
robertaSentiment = transformers.pipeline("sentiment-analysis",model="pdelobelle/robbert-v2-dutch-base")

# function to get all tweets that are not labeled in the database and label them
def update_sentiment(sentiment_model, vectorizer):
    session = Session()
    posts = session.execute(select(Post).where(Post.sentiment == None))
    for row in posts:
        vector = vectorizer.transform([row[0].content])
        sentiment = sentiment_model.predict(vector)
        row[0].sentiment = int(sentiment[0])
        # if sentiment is positive check it against different model 
        # if the tweet is negative in that model delete the tweet
        if sentiment == 1:
            roberta_result = robertaSentiment(row[0].content)[0]['label']
            if roberta_result == "LABEL_0":
                session.query(Post).filter(Post.id==row[0].id).delete()
            else:
                session.add(row[0])
        else:
            session.add(row[0])
        session.commit()

def get_all():
    df = pd.read_sql_query(
    sql = sqlalchemy.select([Post.content,
                     Post.sentiment]),
    con = engine)
    return df