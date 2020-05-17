import logging
import pickle
import sys

import numpy as np
from sortedcontainers import SortedList

USER_LIMIT = 10000


def load_data():
    result = []
    for fname in ['user2movie.pkl', 'movie2user.pkl', 'usermovie2rating.pkl', 'usermovie2rating_test.pkl']:
        with open(fname, 'rb') as f:
            result += [pickle.load(f)]

    return result


def mse(p, t):
    p = np.array(p)
    t = np.array(t)
    return np.mean((p - t)**2)


def user_based():
    user2movie, movie2user, usermovie2rating, usermovie2rating_test = load_data()

    def _predict(i, m):
        numerator = 0
        denominator = 0
        for neg_w, j in neighbors[i]:
            try:
                numerator += -neg_w * deviations[j][m]  # remember, the weight is stored as its negative
                denominator += abs(neg_w)
            except KeyError:
                pass

        if denominator == 0:
            prediction = averages[i]
        else:
            prediction = numerator / denominator + averages[i]
        prediction = min(5, prediction)
        prediction = max(0.5, prediction)  # min rating is 0.5
        return prediction

    num_users = len(user2movie.keys())

    if num_users >= USER_LIMIT:
        logging.warn(f'user limit exceeded: {len(user2movie.keys())}')
        sys.exit(1)

    num_neighbors = 25  # number of neighbors we'd like to consider
    limit = 5  # number of common movies users must have in common in order to consider
    neighbors = []  # store neighbors in this list
    averages = []  # each user's average rating for later use
    deviations = []  # each user's deviation for later use
    for i in range(num_users):
        # find the 25 closest users to user i
        movies_i = user2movie[i]
        movies_i_set = set(movies_i)

        # calculate avg and deviation
        ratings_i = {movie: usermovie2rating[(i, movie)] for movie in movies_i}
        avg_i = np.mean(list(ratings_i.values()))
        dev_i = {movie: (rating - avg_i) for movie, rating in ratings_i.items()}
        dev_i_values = np.array(list(dev_i.values()))
        sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

        # save these for later use
        averages.append(avg_i)
        deviations.append(dev_i)

        sl = SortedList()
        for j in range(num_users):
            if j == i:  # don't include yourself
                continue
            movies_j = user2movie[j]
            movies_j_set = set(movies_j)
            common_movies = (movies_i_set & movies_j_set)  # intersection
            if len(common_movies) > limit:
                # calculate avg and deviation
                ratings_j = {movie: usermovie2rating[(j, movie)] for movie in movies_j}
                avg_j = np.mean(list(ratings_j.values()))
                dev_j = {movie: (rating - avg_j) for movie, rating in ratings_j.items()}
                dev_j_values = np.array(list(dev_j.values()))
                sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

                # calculate correlation coefficient
                numerator = sum(dev_i[m] * dev_j[m] for m in common_movies)
                w_ij = numerator / (sigma_i * sigma_j)

                # insert into sorted list and truncate
                # negate weight, because list is sorted ascending
                # maximum value (1) is "closest"
                sl.add((-w_ij, j))
            if len(sl) > num_neighbors:
                del sl[-1]

        # store the neighbors
        neighbors.append(sl)

        # print out useful things
        if i % 1 == 0:
            print(i)

    # Train
    train_predictions = []
    train_targets = []
    for (i, m), target in usermovie2rating.items():
        # calculate the prediction for this movie
        prediction = _predict(i, m)

        # save the prediction and target
        train_predictions.append(prediction)
        train_targets.append(target)

    # Test
    test_predictions = []
    test_targets = []
    for (i, m), target in usermovie2rating_test.items():
        prediction = _predict(i, m)

        test_predictions.append(prediction)
        test_targets.append(target)

    print('train mse:', mse(train_predictions, train_targets))
    print('test mse:', mse(test_predictions, test_targets))


if __name__ == "__main__":
    user_based()
