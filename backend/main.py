import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_csv("backend/data/movies_metadata.csv", low_memory=False)

#print(data.head())

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

top_ten_rated_movies = q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)

#plot based recommender

tfidf = TfidfVectorizer(stop_words="english")

data["overview"] = data["overview"].fillna("")

tfidf_matrix = tfidf.fit_transform(data["overview"])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(data.index, index=data["title"]).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim):

    print(title)
    
    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return data["title"].groupby(data["title"].iloc[movie_indices]).head(10).tolist()
