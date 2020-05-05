import glob
import os

import pandas as pd

PATH_MOVIELENS = f'{os.environ["HOME"]}/Documents/Datasets/movielens-20m-dataset'
FILES = glob.glob(PATH_MOVIELENS + '/**')


def main():
    assert len(FILES) == 6, f"Number of files should be 6: {len(FILES)}"
    assert os.path.exists(PATH_MOVIELENS)

    df = pd.read_csv(PATH_MOVIELENS + '/rating.csv')

    # Modify userId
    df.userId = df.userId - 1

    # Incremental index for movieId
    movie2idx = {v: i for i, v in enumerate(set(df.movieId.unique()))}
    df.movieId = df.movieId.map(movie2idx)

    # Drop redundant column
    df.drop(columns='timestamp', errors='ignore')

    # Saving
    df.to_csv(PATH_MOVIELENS + '/edited_rating.csv', index=False)


if __name__ == '__main__':
    main()
