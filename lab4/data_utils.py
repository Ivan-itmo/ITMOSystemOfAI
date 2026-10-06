import numpy as np
import pandas as pd


def load_dataset(path):
    df = pd.read_csv(path)
    df.columns = [column.strip() for column in df.columns]
    return df


def encode_categorical_features(df, target_column):
    features = df.drop(columns=[target_column]).copy()
    categorical = features.select_dtypes(exclude=[np.number]).columns.tolist()

    for column in categorical:
        features[column] = features[column].fillna("Unknown")

    if categorical:
        # Каждую категорию представляем отдельным бинарным признаком.
        features = pd.get_dummies(features, columns=categorical, dtype=float)

    return features, categorical


def stratified_split(y, test_size=0.25, seed=42):
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []

    # Делим каждый класс отдельно, сохраняя примерно исходные доли классов.
    for label in np.unique(y):
        indices = np.where(y == label)[0]
        rng.shuffle(indices)
        test_count = max(1, int(round(len(indices) * test_size)))
        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])

    train_indices = np.array(train_indices, dtype=int)
    test_indices = np.array(test_indices, dtype=int)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return train_indices, test_indices


def impute_numeric(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    for column in train_df.columns:
        train_df[column] = pd.to_numeric(train_df[column], errors="coerce")
        test_df[column] = pd.to_numeric(test_df[column], errors="coerce")
        # Пропуски в обеих выборках заполняем медианой обучающей выборки.
        median = train_df[column].median()
        train_df[column] = train_df[column].fillna(median)
        test_df[column] = test_df[column].fillna(median)

    return train_df, test_df


def standardize(train_df, test_df):
    # Параметры масштаба вычисляем только на обучающих данных.
    mean = train_df.mean(axis=0)
    std = train_df.std(axis=0, ddof=0).replace(0, 1)
    train_scaled = (train_df - mean) / std
    test_scaled = (test_df - mean) / std
    return train_scaled, test_scaled
