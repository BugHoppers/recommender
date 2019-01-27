import numpy as np
import pandas as pd


def get_ratings(csv):
    ratings = pd.read_csv(csv, sep=',', encoding='latin-1',
                          usecols=['userId', 'movieId', 'rating'])
    max_userid = ratings['userId'].drop_duplicates().max()
    max_movieid = ratings['movieId'].drop_duplicates().max()
    return ratings, max_userid, max_movieid


def get_movies(csv):
    movies = pd.read_csv(csv, sep=',', encoding='latin-1',
                         usecols=['movieId', 'title', 'genres'])
    return movies


def create_train_set(csv):
    ratings, _, _ = get_ratings(csv)
    shuffled_ratings = ratings.sample(frac=1., random_state=np.random.RandomState)
    return shuffled_ratings


def shuffleData(csv):
    shuffled_ratings = create_train_set(csv)
    Users = shuffled_ratings['userId'].values
    Movies = shuffled_ratings['movieId'].values
    Ratings = shuffled_ratings['rating'].values
    return Users, Movies, Ratings
