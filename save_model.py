# save_model.py
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

def convert(obj):
    try:
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L
    except:
        return []

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])
movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'])

movies['overview'] = movies['overview'].fillna('')
movies['tags'] = movies['overview'] + movies['genres'].apply(lambda x: " ".join(x)) + \
                 movies['keywords'].apply(lambda x: " ".join(x)) + \
                 movies['cast'].apply(lambda x: " ".join(x)) + \
                 movies['crew'].apply(lambda x: " ".join(x))
movies['tags'] = movies['tags'].apply(lambda x: x.lower())

new_df = movies[['movie_id', 'title', 'tags']]
tfidf = TfidfVectorizer(stop_words='english')
vector = tfidf.fit_transform(new_df['tags'].values.astype('U'))
similarity = cosine_similarity(vector)

# Save files
pickle.dump(new_df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
