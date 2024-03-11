import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD  # Singular Value Decomposition

def GenerateUserMovieRating():
    # Reading CSV files into dataframs
    movies_raw = pd.read_csv('ml-latest-small/movies.csv')
    rating_raw = pd.read_csv('ml-latest-small/ratings.csv')

    # print(movies_raw.values.shape)
    # print(movies_raw.head(10))

    # print(rating_raw.values.shape)
    # print(rating_raw.head(10))

    rated_movies_raw = pd.merge(rating_raw, movies_raw, on='movieId')

    # print(rated_movies_raw.values.shape)
    # print(rated_movies_raw.head(10))

    rated_movies_no_ts_or_genre = rated_movies_raw.drop(['timestamp', 'genres'], axis=1)

    # print(rated_movies_no_ts_or_genre.values.shape)
    # print(rated_movies_no_ts_or_genre.head(10))

    rated_movies = rated_movies_no_ts_or_genre.dropna(axis=0, subset=['title'])    # Remove any entries that don't have a title for some reason
    # print(rated_movies)

    rated_movies_count = rated_movies.groupby(by = ['title'])['rating'].count().reset_index().rename(columns={'rating': 'totalRatingCount'})[['title', 'totalRatingCount']]
    rating_with_totalRatingCount = rated_movies.merge(rated_movies_count, left_on='title', right_on='title', how='left')
    rating_with_totalRatingCount.head()

    # print(rating_with_totalRatingCount.values.shape)
    # print(rating_with_totalRatingCount.head())

    user_rating = rating_with_totalRatingCount.drop_duplicates(['userId', 'title'])

    # print(user_rating.values.shape)
    # print(user_rating.head(10))

    user_movie_rating = user_rating.pivot(index='userId', columns='title', values='rating').fillna(0)
    # print(user_movie_rating.values.shape)
    # print(user_movie_rating.head(10))
    return user_movie_rating

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    P     : an initial matrix of dimension N x K
    Q     : an initial matrix of dimension M x K
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    for step in range(steps):
        print(step)
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    for k in range(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    return P, Q.T

if __name__ == "__main__":
    movie_ratings = GenerateUserMovieRating()
    print(movie_ratings.head())
    movie_titles = movie_ratings.columns.to_list()

    R = np.array(movie_ratings)
    N = len(R)
    M = len(R[0])
    K = 2

    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)

    nP, nQ = matrix_factorization(R, P, Q, K, steps=1000)
    nR = np.dot(nP, nQ.T)
    nR_df = pd.DataFrame(nR, columns=movie_titles)
    print(nR_df.head())

    nR_df.to_csv('RatingsModel.csv', index=False)