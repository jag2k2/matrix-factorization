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

    # print(rated_movies_no_ts_or_genre.shape)
    # print(rated_movies_no_ts_or_genre.values.shape)

    rated_movies = rated_movies_no_ts_or_genre.dropna(axis=0, subset=['title'])    # Remove any entries that don't have a title for some reason
    # print(rated_movies)
    # print(rated_movies.shape)

    rated_movies_count = rated_movies.groupby(by = ['title'])['rating'].count().reset_index().rename(columns={'rating': 'totalRatingCount'})[['title', 'totalRatingCount']]
    rating_with_totalRatingCount = rated_movies.merge(rated_movies_count, left_on='title', right_on='title', how='left')

    # print(rating_with_totalRatingCount.values.shape)
    # print(rating_with_totalRatingCount.head())

    user_rating = rating_with_totalRatingCount.drop_duplicates(['userId', 'title'])

    # print(user_rating.values.shape)
    # print(user_rating.head(10))

    user_movie_rating = user_rating.pivot(index='userId', columns='title', values='rating').fillna(0)
    # # print(user_movie_rating.values.shape)
    # print(user_movie_rating.head(10))
    return user_movie_rating

"""
@INPUT:
    N : Number of users. This input is implied within the dimensions of R.
    
    M : Number of movies. This input is implied within the dimensions of R.

    R <dimension N x M> : The matrix to be factorized. Contains all the ratings that users have 
    assigned to the movies. Initially, this matrix would contain only the ratings that each
    user has assigned, but after factorization, it contains the predictions as well as 
    approximations to the original rating each user gave to the movies they actually rated.

    P <dimension N x K> : Strength associations between a user and the underlying features 

    Q <dimension M x K> : Strength associations between a movie and the underlying features
    
    K     : Number of latent features. This number tells the algorithm how many features it
    should associate users and items to for predicting the missing ratings of a user in R
    
    steps : the maximum number of steps to perform the optimization

    alpha : the learning rate for the gradient descent during the optimization

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

    movie_ratings.to_csv("movieRatingsGenerated.csv")
    print(movie_ratings.head())
    movie_titles = movie_ratings.columns.to_list()

    R = np.array(movie_ratings)
    N = len(R)
    M = len(R[0])
    K = 2
    kList = [5,10,15]
    for testK in kList: # test
        K = testK
        P = np.random.rand(N,K)
        Q = np.random.rand(M,K)
        Steps = 5000
        nP, nQ = matrix_factorization(R, P, Q, K, steps=Steps)
        nR = np.dot(nP, nQ.T)
        nR_df = pd.DataFrame(nR, columns=movie_titles)
        print(nR_df.head())

        nR_df.to_csv("RatingsModel_k_{}_steps_{}.csv".format(K,Steps), index=False)