import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


data = pd.read_csv("data/movies_metadata.csv")

print(data.head())

#calculate weighted rating

C = data["vote_average"].mean()
m = data["vote_count"].quantile(0.9)

q_movies = data.copy().loc[data["vote_count"] >= m]


def weighted_rating(x, m=m, C=C):
    v = x["vote_count"]
    R = x["vote_average"]
    return (v/(v+m) * R) + (m/(m+v) * C)

q_movies["score"] = q_movies.apply(weighted_rating, axis=1)

q_movies = q_movies.sort_values("score", ascending=False)  

print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10))