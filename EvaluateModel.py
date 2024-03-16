import pandas as pd
import numpy as np
from CreateModel import GenerateUserMovieRating

if __name__ == '__main__':
    user_ratings = GenerateUserMovieRating()
    ratings_model = pd.read_csv('RatingsModel.csv')
    user_movies_dict = {row + 1: user_ratings.columns[user_ratings.iloc[row] > 0].tolist() for row in range(len(user_ratings))}
    
    count = 0
    square_sum = 0
    for user, movies in user_movies_dict.items():
        for movie in movies:
            count += 1
            difference = user_ratings.at[user, movie] - ratings_model.at[user-1, movie]         # userIDs are off by one
            square_sum += (difference) ** 2

    mse = square_sum / count
    print(mse)