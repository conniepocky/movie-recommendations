import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval

movies = pd.read_csv("backend/data/movies_metadata.csv", low_memory=False)
keywords = pd.read_csv("backend/data/keywords.csv", low_memory=False)

movies.drop(columns=["imdb_id", "budget", "homepage", "status", "video"])

keywords["id"] = keywords["id"].astype(str)

data = movies.merge(keywords, on="id")

print(data.head(5))

def get_movie_id_by_title(title):
    if data.loc[data["title"] == title].empty:
        return "error"
    else:
        return data.loc[data["title"] == title]["id"].tolist()[0]
    
def get_movie_title_by_id(id):
    return data.loc[data["id"] == id]["title"].tolist()[0]

def get_movie_image_by_id(id): #todo fix this
    isCollection = data.loc[data["id"] == id].belongs_to_collection.values
    print(isCollection)
    if pd.isna(isCollection):
        poster_path = data.loc[data["id"] == id]["poster_path"].tolist()[0]
    else:
        d = eval(data.loc[data["id"] == id].belongs_to_collection.values[0])
        print(d)
        poster_path = d.get("poster_path")
    return "https://image.tmdb.org/t/p/w185" + str(poster_path)

print(get_movie_image_by_id("862"))

#calculate weighted rating (imbd formula). demographical filtering

C = data["vote_average"].mean()
m = data["vote_count"].quantile(0.9)

q_movies = data.copy().loc[data["vote_count"] >= m]


def weighted_rating(x, m=m, C=C):
    v = x["vote_count"]
    R = x["vote_average"]
    return (v/(v+m) * R) + (m/(m+v) * C)

q_movies["score"] = q_movies.apply(weighted_rating, axis=1)

q_movies = q_movies.sort_values("score", ascending=False)  

top_ten_rated_movies = q_movies[['title', 'vote_count', 'vote_average', 'score', "overview"]].head(10)

def get_best_movies():
    titles = top_ten_rated_movies["title"].tolist()
    scores = top_ten_rated_movies["score"].tolist()

    scores = [round(num, 2) for num in scores]

    overview = top_ten_rated_movies["overview"].tolist()

    
    return [list(row) for row in zip(titles, scores, overview)]

#plot based recommender

def get_recommendations(id, cosine_sim):

    title = get_movie_title_by_id(id)
    
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    try:
        return data["title"].groupby(data["title"].iloc[movie_indices]).head(10).tolist()
    except:
        return ["No recommendations found"]

#recommender
    
stemmer = SnowballStemmer("english")
    
features = ["genres", "keywords", "adult"]

for feature in features:
    data[feature] = data[feature].apply(literal_eval)

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names
    
    return []

for feature in ["genres", "keywords"]:
    data[feature] = data[feature].apply(get_list)

data['keywords'] = data['keywords'].apply(lambda x: [stemmer.stem(i) for i in x]) #stem keywords

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ""
        
for feature in features:
    data[feature] = data[feature].apply(clean_data)

def create_soup(x):
    return " ".join(x["genres"]) + " ".join(x["keywords"]) + " " + x["adult"]

data["soup"] = data.apply(create_soup, axis=1)

tfidif = TfidfVectorizer(stop_words="english")

count_matrix = tfidif.fit_transform(data["soup"])

cosine_sim2 = linear_kernel(count_matrix, count_matrix)

data = data.reset_index()

indices = pd.Series(data.index, index=data["title"])

print(get_recommendations("65", cosine_sim2))