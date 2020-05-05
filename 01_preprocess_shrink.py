import os
import pandas as pd

N_USERS_TO_KEEP = 1000
N_MOVIES_TO_KEEP = 2000

PATH_MOVIELENS = f'{os.environ["HOME"]}/Documents/Datasets/movielens-20m-dataset'


def main():
    df = pd.read_csv(PATH_MOVIELENS + "/edited_rating.csv")
    print(f"edited_rating.csv shape: {df.shape}")

    user_ids = df.userId.value_counts()[:N_USERS_TO_KEEP].index
    movie_ids = df.movieId.value_counts()[:N_MOVIES_TO_KEEP].index

    # sanity check
    assert user_ids.shape[0] == N_USERS_TO_KEEP
    assert movie_ids.shape[0] == N_MOVIES_TO_KEEP

    # make a copy, otherwise ids won't be overwritten
    df_small = df[df.userId.isin(user_ids) & df.movieId.isin(movie_ids)].copy()

    user_id_map = {v: i for i, v in enumerate(user_ids)}
    movie_id_map = {v: i for i, v in enumerate(movie_ids)}

    df_small.movieId = df_small.movieId.map(movie_id_map)
    df_small.userId = df_small.userId.replace(user_id_map)

    # sanity check
    assert not df_small.movieId.isna().max(), 'check previous map fuctions'
    assert not df_small.userId.isna().max(), 'check previous map fuctions'

    print(f"small dataframe size: {df_small.shape[0]}")

    # Saving
    df_small.to_csv(PATH_MOVIELENS + '/small_rating.csv', index=False)


if __name__ == "__main__":
    main()
