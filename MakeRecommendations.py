import pandas as pd
import numpy as np
from CreateModel import GenerateUserMovieRating

if __name__ == '__main__':
    user_id = 1
    number_of_recommendations = 5

    user_ratings = GenerateUserMovieRating()
    user_movies_dict = {row + 1: user_ratings.columns[user_ratings.iloc[row] > 0].tolist() for row in range(len(user_ratings))}

    ratings_model = pd.read_csv('RatingsModel.csv')
    user_ratings_model = ratings_model.iloc[user_id - 1]
    sorted_user_ratings = user_ratings_model.sort_values(ascending=False)
    print(sorted_user_ratings)
    recommendations = []
    for name, value in sorted_user_ratings.items():
        if name not in user_movies_dict.get(user_id, []):
            recommendations.append(name)
        if len(recommendations) >= number_of_recommendations:
            break
    print("Based on your movie ratings, we think you will also like the following movies:")
    print(recommendations)