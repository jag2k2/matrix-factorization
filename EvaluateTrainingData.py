from CreateModel import GenerateUserMovieRating
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    movies_raw = pd.read_csv('ml-latest-small/movies.csv')
    rating_raw = pd.read_csv('ml-latest-small/ratings.csv')
    movie_ratings = GenerateUserMovieRating()
    print(rating_raw.head())
    print(rating_raw.shape[0], "ratings")
    print(len(rating_raw['userId'].unique()), "users")
    print(len(rating_raw['movieId'].unique()), "movies")

    # plt.hist(rating_raw['rating'], bins=5)
    # plt.xlabel('Rating')
    # plt.ylabel('Frequency')
    # plt.title('Rating Distribution')
    # plt.grid()
    # plt.show()

    # ratings_per_user = rating_raw.groupby('userId').size()
    # plt.hist(ratings_per_user, bins=30)
    # plt.xlabel('Number of Ratings')
    # plt.ylabel('Number of Users')
    # plt.title('Ratings per User')
    # plt.grid()
    # plt.show() 

    average_rating_per_user = rating_raw.groupby('userId')['rating'].mean()

    # Now, plot the histogram
    plt.hist(average_rating_per_user, bins=30)

    # Add labels and title
    plt.xlabel('Average Rating')
    plt.ylabel('Number of Users')
    plt.title('Average Ratings per User')
    plt.grid()
    plt.show()