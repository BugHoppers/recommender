import numpy as np
import pandas as pd


def get_ratings(csv_rating):
    ratings = pd.read_csv(csv_rating, sep='\t', encoding='latin-1',
                          usecols=['user_id', 'movie_id', 'user_emb_id', 'movie_emb_id', 'rating'])
    max_userid = ratings['user_id'].drop_duplicates().max()
    max_movieid = ratings['movie_id'].drop_duplicates().max()
    return ratings, max_userid, max_movieid


def get_users(csv_users):
    users = pd.read_csv(csv_users, sep='\t', encoding='latin-1',
                        usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])
    return users


def get_movies(csv_movies):
    movies = pd.read_csv(csv_movies, sep='\t', encoding='latin-1',
                         usecols=['movie_id', 'title', 'genres'])
    return movies


def create_train_set():
    ratings = get_ratings('ratings.csv')
    shuffled_ratings = ratings.sample(frac=1., random_state=RNG_SEED)
    return shuffled_ratings


def shuffle():
    shuffled_ratings = create_train_set()
    Users = shuffled_ratings['user_emb_id'].values
    Movies = shuffled_ratings['movie_emb_id'].values
    Ratings = shuffled_ratings['rating'].values
    return Users, Movies, Ratings
