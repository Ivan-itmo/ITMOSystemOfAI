from pathlib import Path
import os
import tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.gettempdir())
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TARGET = 'median_house_value'
SYNTHETIC = ['rooms_per_household', 'bedrooms_per_room', 'population_per_household']


def load_and_visualize(filepath, output_dir='results'):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(filepath)
    if TARGET not in df:
        raise ValueError(f'Не найден целевой столбец {TARGET}.')
    statistics = df.describe(percentiles=[.05, .25, .5, .75, .95]).T
    statistics.to_csv(output / 'statistics.csv', index_label='feature')
    missing = df.isna().sum().rename('missing')
    missing.to_csv(output / 'missing_values.csv', index_label='feature')
    print('Описательная статистика:\n', statistics.round(2).to_string())
    print('\nПропуски:\n', missing.to_string())
    numeric = df.select_dtypes(include='number')
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    for ax, col in zip(axes.ravel(), numeric.columns):
        ax.hist(df[col].dropna(), bins=50, color='#4477aa', edgecolor='white')
        ax.axvline(df[col].mean(), color='#cc3311', label='Mean')
        ax.set_title(col)
        ax.set_ylabel('Count')
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'histograms.png', dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(17, 5))
    ax.axis('off')
    table = ax.table(cellText=statistics.round(2).astype(str).values,
                     rowLabels=statistics.index, colLabels=statistics.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    ax.set_title('Статистика', pad=25)
    fig.tight_layout()
    fig.savefig(output / 'statistics.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    corr = numeric.corr()
    corr.to_csv(output / 'correlations.csv')
    fig, ax = plt.subplots(figsize=(11, 9))
    heatmap = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(heatmap, ax=ax)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            value = corr.iloc[i, j]
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                    color='white' if abs(value) > .5 else 'black')
    ax.set_title('Корреляция Пирсона')
    fig.tight_layout()
    fig.savefig(output / 'correlation_matrix.png', dpi=150)
    plt.close(fig)
    return df


def split_dataframe(df, test_size=.2, random_state=42):
    if not 0 < test_size < 1 or len(df) < 2:
        raise ValueError('Нужны минимум две строки и 0 < test_size < 1.')
    indices = np.random.default_rng(random_state).permutation(len(df))
    split = min(len(df) - 1, max(1, int(len(df) * (1 - test_size))))
    return df.iloc[indices[:split]].copy(), df.iloc[indices[split:]].copy()


def add_synthetic_features(df):
    df = df.copy()
    for name, numerator, denominator in [
        ('rooms_per_household', 'total_rooms', 'households'),
        ('bedrooms_per_room', 'total_bedrooms', 'total_rooms'),
        ('population_per_household', 'population', 'households'),
    ]:
        df[name] = df[numerator] / df[denominator].replace(0, np.nan)
    return df.replace([np.inf, -np.inf], np.nan)


class DataPreprocessor:
    def fit(self, train):
        self.numeric = list(train.select_dtypes(include='number').columns)
        self.categorical = [c for c in train if c not in self.numeric]
        self.fill_values = train[self.numeric].replace([np.inf, -np.inf], np.nan).mean().fillna(0)
        self.categories = {}
        for col in self.categorical:
            values = train[col].astype('string').fillna('<missing>')
            self.categories[col] = sorted(values.unique().tolist())
        encoded = self._encode(train)
        self.feature_names = list(encoded.columns)
        self.mean = encoded.mean()
        self.std = encoded.std(ddof=0).replace(0, 1).fillna(1)
        return self

    def _encode(self, df):
        encoded = df[self.numeric].replace([np.inf, -np.inf], np.nan).fillna(self.fill_values).astype(float)
        for col, categories in self.categories.items():
            values = df[col].astype('string').fillna('<missing>')
            for i, category in enumerate(categories[1:], start=1):
                encoded[f'{col}__{i}={category}'] = (values == category).astype(float)
        return encoded

    def transform(self, df):
        if not hasattr(self, 'mean'):
            raise RuntimeError('Сначала вызовите fit().')
        scaled = ((self._encode(df) - self.mean) / self.std).to_numpy(dtype=float)
        return np.column_stack((np.ones(len(df)), scaled))


def prepare_features(train, test, feature_cols, target_col=TARGET):
    processor = DataPreprocessor().fit(train[feature_cols])
    return (processor.transform(train[feature_cols]), processor.transform(test[feature_cols]),
            train[target_col].to_numpy(float), test[target_col].to_numpy(float), processor)
