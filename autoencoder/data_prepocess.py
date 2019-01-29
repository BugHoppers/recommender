# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

def convert(data, n_users, n_movies):
    new_data = []
    for id_users in range(1, n_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(n_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

def load_data(csv):
    dataset = pd.read_csv(csv)
    dataset = dataset.iloc[:,:-1].astype(np.int64).values
    training_set, test_set = train_test_split(dataset, test_size=0.2)
    
    n_users = int(max(max(training_set[:,0]), max(test_set[:,0])))
    n_movies = int(max(max(training_set[:,1]), max(test_set[:,1])))
    
    training_set = convert(training_set, n_users, n_movies)
    test_set = convert(test_set, n_users, n_movies)
    
    training_set = torch.FloatTensor(training_set)
    test_set = torch.FloatTensor(test_set)
    
    return training_set, test_set, n_users, n_movies