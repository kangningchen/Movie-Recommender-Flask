import pandas as pd
import numpy as np
import json
import requests, urllib
from os import mkdir, listdir
from bs4 import BeautifulSoup  
from tqdm import tqdm
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


DATASET_FOLDER = 'the-movies-dataset'
IMG_FOLDER = 'img/'
CACHE_FNAME = 'poster_urls.json'

# get cache file
try:
    cache_file = open(CACHE_FNAME, 'r')
    CACHE_DICTION = json.loads(cache_file.read())
    cache_file.close()
except:
    CACHE_DICTION = {}

# load movies_metadata.csv, drop unused columns, and construct imdb urls
def load_data():
    movies_metadata = pd.read_csv(DATASET_FOLDER + '/movies_metadata.csv')
    required_columns = ['imdb_id', 'original_title', 'overview']
    movies_metadata = movies_metadata[required_columns]
    movies_metadata = movies_metadata[:1000]
    movies_metadata['imdb_url'] = 'https://www.imdb.com/title/' + movies_metadata['imdb_id']
    return movies_metadata

# crawl and cache poster urls
def get_poster_urls(movies_metadata):
    for url in tqdm(movies_metadata['imdb_url']):
        if url not in CACHE_DICTION:
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')  
            results = soup.find_all('div', attrs={'class':'poster'}) 
            if len(results) == 0: 
                poster_url = ""
            else: 
                first_result = results[0] 
                poster_url = first_result.find('img')['src'] 
                CACHE_DICTION[url] = poster_url
                dumped_json_cache = json.dumps(CACHE_DICTION)
                f = open(CACHE_FNAME,'w')
                f.write(dumped_json_cache)
                f.close()  

# read poster urls json, and merge poster urls into movies_metadata.csv
def merge_poster_urls(movies_metadata):
    with open(CACHE_FNAME) as json_data:
        data = json.load(json_data)
    poster_urls = pd.DataFrame.from_dict(data, orient='index').reset_index()
    poster_urls.rename(columns={'index':'imdb_url', 0:'poster_url'}, inplace=True)
    movies_metadata = movies_metadata.merge(poster_urls, on='imdb_url')
    movies_metadata.to_pickle('movies_metadata.pkl')
    return movies_metadata

# download posters programmatically 
def download_posters(movies_metadata):
    try:
        mkdir(IMG_FOLDER)
        for index, row in tqdm(movies_metadata.iterrows()):
            url = row['poster_url']
            if len(url) > 0:
                imbd_id = str(row['imdb_id'])
                jpgname = IMG_FOLDER + imbd_id +'.jpg'
                urllib.request.urlretrieve(url, jpgname)
    except:
        print('''Creation of img folder probably failed. Please manually create 
        a folder named img in the project directory. If folder creation was successful, 
        it could be due to massively requesting 1000 poster images. Wait for a few minutes
        and try running the fuction again.
        ''')

# get Inception V3 model
def get_inceptionv3():
    inception_v3 = InceptionV3(weights='imagenet', include_top=False)
    model= Model(inputs=inception_v3.input,  outputs=inception_v3.get_layer(index=-1).output)
    return model

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

def get_image_feature_array():
    model = get_inceptionv3()
    images = []
    for image_name in listdir(IMG_FOLDER):
        f_path = 'img/' + image_name
        image = load_img(f_path, target_size=(299, 299))
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = preprocess_input(image_array)
        image_id = image_name.split('.')[0]
        feature_array = model.predict(image_array).ravel()
        images.append((image_id, feature_array))
    images_df = pd.DataFrame(images)
    images_df.rename(columns={0:'imdb_id', 1:'feature_array'}, inplace=True)
    images_df.to_pickle('images_df.pkl')
    return images_df

def pickle_df(movies_metadata, images_df):
    df = movies_metadata.merge(images_df, on='imdb_id')
    df.to_pickle('df.pkl')

def main():
    user_input = input('''You are about to crawl 1000 pages and download about 1000 images. 
    Do you want to proceed? [Y/N]''')
    if user_input.upper() == 'Y' or user_input.upper() == 'YES':
        movies_metadata = load_data()
        get_poster_urls(movies_metadata)
        movies_metadata = merge_poster_urls(movies_metadata)
        download_posters(movies_metadata)
        images_df = get_image_feature_array()
        pickle_df(movies_metadata, images_df)
    else:
        break

if __name__=='__main__':
    main()