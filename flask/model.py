import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def load_pickled_df():
    df = pd.read_pickle('static/data/df.pkl')
    return df

def fit_knn(feature_array, nn=6, algorithm='ball_tree', metric='euclidean'):
    knn = NearestNeighbors(n_neighbors=nn, algorithm=algorithm, metric=metric, n_jobs = -1).fit(feature_array) 
    return knn  

def find_nbrs(nbr_indices, images_df):
    nbrs = []
    for i in range(len(nbr_indices)): 
        nbr_index = nbr_indices[i]
        image_id = images_df[(images_df.index == nbr_index)].iloc[0]['imdb_id']
        original_title = images_df[(images_df.index == nbr_index)].iloc[0]['original_title']
        overview = images_df[(images_df.index == nbr_index)].iloc[0]['overview']
        nbrs.append([image_id, original_title, overview])
    return nbrs

def construct_nbr_file_path(image_id, dir):
    path = 'static/' + dir + image_id + '.jpg'
    return path


def show_recommendations(original_title, df, dir='img/', feature_column_name='feature_array'):
    knn = fit_knn(list(df['feature_array']))
    feature_array = df[(df['original_title'].str.lower() == original_title.lower())].iloc[0][feature_column_name].reshape(1,-1)
    nbr_indices = knn.kneighbors(feature_array, return_distance=False)
    nbr_indices = nbr_indices.tolist()[0]
    nbrs = find_nbrs(nbr_indices, df)
    for nbr in nbrs:
        path = construct_nbr_file_path(nbr[0], dir)
        nbr.append(path)
    return nbrs





