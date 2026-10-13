from pathlib import Path

import numpy as np
import pandas as pd


COLUMNS = [
    "class",
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
]


def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS, dtype=str)
    return df.drop_duplicates().reset_index(drop=True)


def stratified_train_test_split(df: pd.DataFrame, target: str, test_size: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []

    for _, group in df.groupby(target, sort=False):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        test_count = max(1, int(round(len(indices) * test_size)))
        test_parts.append(df.loc[indices[:test_count]])
        train_parts.append(df.loc[indices[test_count:]])

    train = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, test


class CategoricalPreprocessor:
    def __init__(self):
        self.fill_values: dict[str, str] = {}

    def fit(self, X: pd.DataFrame):
        for column in X.columns:
            valid = X.loc[X[column] != "?", column]
            self.fill_values[column] = valid.mode().iloc[0] if not valid.empty else "missing"
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        for column, fill_value in self.fill_values.items():
            result[column] = result[column].replace("?", fill_value).fillna(fill_value).astype(str)
        return result


def select_random_sqrt_features(feature_names: list[str], seed: int = 10) -> list[str]:
    # округляем корень вниз и выбираем признаки без повторений
    count = max(1, int(np.sqrt(len(feature_names))))
    rng = np.random.default_rng(seed)
    return [str(x) for x in rng.choice(feature_names, size=count, replace=False)]
