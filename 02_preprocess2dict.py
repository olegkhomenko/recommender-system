import os
import pickle
from typing import Dict

import pandas as pd
from sklearn.model_selection import train_test_split

import constants as C


def update_user2movie_and_movie2user(df: pd.DataFrame) -> Dict[str, dict]:
    # TODO: Try to fill up dictionaries in parallel using parrallel_apply() instead of itertuples
    user2movie = {}
    movie2user = {}
    usermovie2rating = {}

    for row in df.itertuples():
        i = row.userId
        j = row.movieId
        if i not in user2movie:
            user2movie[i] = [j]
        else:
            user2movie[i].append(j)

        if j not in movie2user:
            movie2user[j] = [i]
        else:
            movie2user[j].append(i)

        usermovie2rating[(i, j)] = row.rating

    return dict(user2movie=user2movie, movie2user=movie2user, usermovie2rating=usermovie2rating)


def update_usermovie2rating_test(df: pd.DataFrame) -> Dict[str, dict]:
    usermovie2rating_test = {}
    for row in df.itertuples():
        i = row.userId
        j = row.movieId
        usermovie2rating_test[(i, j)] = row.rating

    return dict(usermovie2rating_test=usermovie2rating_test)


def main():
    df = pd.read_csv(os.path.join(C.PATH_MOVIELENS, 'small_rating.csv'))
    df_train, df_test = train_test_split(df, train_size=C.TRAIN_SIZE)

    # Train
    dicts = update_user2movie_and_movie2user(df_train)

    # Test
    dict_test = update_usermovie2rating_test(df_test)

    objects_to_save = {**dicts, **dict_test}
    # 'user2movie.pkl', 'movie2user.pkl', 'usermovie2rating.pkl', 'usermovie2rating_test.pkl'
    for name, obj in objects_to_save.items():
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f)


if __name__ == "__main__":
    main()
