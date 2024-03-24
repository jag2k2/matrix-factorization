############################################################################
# magomez4 - 3_23_24 - WARNING: THIS FILE IS DEPRECATED. FIXES FOR THE BUGS
# IN THIS ORIGINAL CODEBASE HAVE BEEN MADE IN THE JUPITER NOTEBOOK WITH THE
# SAME NAME "pypiModel.ipynb"
###########################################################################
from matrix_factorization import BaselineModel, KernelMF, train_update_test_split

import pandas as pd
from sklearn.metrics import mean_squared_error

# Movie data found here https://grouplens.org/datasets/movielens/
# cols = ["user_id", "item_id", "rating", "timestamp"]
# movie_data = pd.read_csv(
#     "../data/ml-100k/u.data", names=cols, sep="\t", usecols=[0, 1, 2], engine="python"
# )


# Adaptation for ml-latest-small version of dataset
cols = ["userId", "movieId", "rating", "timestamp"]
movie_data = pd.read_csv(
    "../ml-latest-small/ratings.csv", names=cols, sep="\t", usecols=[0, 1, 2], engine="python"
)


X = movie_data[["userId", "movieId"]]
y = movie_data["rating"]

# Prepare data for online learning
(
    X_train_initial,
    y_train_initial,
    X_train_update,
    y_train_update,
    X_test_update,
    y_test_update,
) = train_update_test_split(movie_data, frac_new_users=0.2)

# Initial training
matrix_fact = KernelMF(n_epochs=20, n_factors=100, verbose=1, lr=0.001, reg=0.005)
matrix_fact.fit(X_train_initial, y_train_initial)

# Update model with new users
matrix_fact.update_users(
    X_train_update, y_train_update, lr=0.001, n_epochs=20, verbose=1
)
pred = matrix_fact.predict(X_test_update)
rmse = mean_squared_error(y_test_update, pred, squared=False)
print(f"\nTest RMSE: {rmse:.4f}")

# Get recommendations
user = 200
items_known = X_train_initial.query("userId == @user")["movieId"]
matrix_fact.recommend(user=user, items_known=items_known)