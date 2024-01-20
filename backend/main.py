import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_csv("backend/data/movies_metadata.csv", low_memory=False)

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

tfidf = TfidfVectorizer(stop_words="english")

data["overview"] = data["overview"].fillna("")

tfidf_matrix = tfidf.fit_transform(data["overview"])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(data.index, index=data["title"]).drop_duplicates()

def get_recommendations(id, cosine_sim=cosine_sim):

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
