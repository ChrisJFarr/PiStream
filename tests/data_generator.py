import pandas as pd
import numpy as np

# Generate synthetic test data


def generate_data(unique_user_n=100, unique_prod_n=10):

    def get_rating(user_id, prod_id):
        max_score = 100
        # Produces random rating, user_id closer to prod_id (relative to range)
        # averages higher. This is the relationship the model needs to pick up in tests
        # Calculate the distance between the user_id relative to all and prod_id relative to all prod_id
        similarity_score = 1 - abs((user_id / (unique_user_n - 1)) - (prod_id / (unique_prod_n - 1)))
        average = int(similarity_score * max_score)
        rating = int(min(max_score, max(1, np.random.normal(loc=average, scale=1))))  # Increase scale for challenge
        return rating

    user_list = range(unique_user_n)
    product_list = range(unique_prod_n)
    test_data = pd.DataFrame(columns=["user_id", "prod_id", "rating"])

    for user_id in user_list:
        # Produce a random number of ratings
        rating_count = np.random.randint(2, len(product_list) - 2)
        # Sample products randomly without repeats
        prod_list = np.random.choice(product_list, size=rating_count, replace=False)
        ratings = [get_rating(user_id=user_id, prod_id=prod_id) for prod_id in prod_list]
        temp = pd.DataFrame(
            columns=["user_id", "prod_id", "rating"],
            data=[[user_id, prod_id, rating] for prod_id, rating in zip(prod_list, ratings)])
        test_data = test_data.append(temp)

    test_data.sort_values(by=list(test_data.columns), inplace=True)
    test_data.reset_index(drop=True, inplace=True)
    test_data = test_data.astype(np.int64)
    return test_data


test_data = generate_data()


