import pandas as pd
import numpy as np
from CreateModel import GenerateUserMovieRating
import matplotlib.pyplot as plt

if __name__ == '__main__':
    user_ratings = GenerateUserMovieRating()
    ks = [2, 4, 8, 16]
    step_list = [10, 100, 1000, 5000]
    for k in ks:
        mse_list = []
        for steps in step_list:
            file_path = f'RatingsModels/RatingsModel_{k}_{steps}.csv'
            ratings_model = pd.read_csv(file_path)
            user_movies_dict = {row + 1: user_ratings.columns[user_ratings.iloc[row] > 0].tolist() for row in range(len(user_ratings))}
            count = 0
            square_sum = 0
            for user, movies in user_movies_dict.items():
                for movie in movies:
                    count += 1
                    difference = user_ratings.at[user, movie] - ratings_model.at[user-1, movie]         # userIDs are off by one
                    square_sum += (difference) ** 2

            mse = square_sum / count
            mse_list.append(mse)
            print(k, ':', steps, ' - ', mse)
        plt.plot(step_list, mse_list, label=f'k={k}', marker='o')
    plt.xlabel('Number of Steps')
    plt.ylabel('MSE')
    plt.title('MSE of Ratings Model vs User Inputs')
    plt.legend()
    plt.xscale('log')
    plt.grid()
    plt.show()