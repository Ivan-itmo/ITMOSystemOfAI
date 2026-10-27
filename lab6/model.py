"""Логистическая регрессия и предобработка без sklearn."""
import numpy as np
import pandas as pd


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -np.asarray(z, dtype=float)))


def log_loss(y, logits):
    """Устойчивая кросс-энтропия; второй аргумент — логиты."""
    return float(np.mean(np.logaddexp(0, logits) - y * logits))


def metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(accuracy=float(np.mean(y == pred)), precision=p, recall=r,
                f1=2*p*r/(p+r) if p+r else 0.0, tp=tp, tn=tn, fp=fp, fn=fn)


def stratified_split(y, fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    left, right = [], []
    for label in np.unique(y):
        ids = rng.permutation(np.flatnonzero(np.asarray(y) == label))
        n = int(round(len(ids) * fraction))
        right.extend(ids[:n]); left.extend(ids[n:])
    return rng.permutation(left), rng.permutation(right)


class TitanicPreprocessor:
    numeric = ['Age', 'SibSp', 'Parch', 'Fare']
    categories = {'Pclass': [1, 2, 3], 'Sex': ['female', 'male'], 'Embarked': ['C', 'Q', 'S']}

    def _encode(self, frame):
        cols = {c: frame[c].fillna(self.medians[c]).astype(float) for c in self.numeric}
        for col, levels in self.categories.items():
            values = frame[col].fillna(self.modes[col])
            if not values.isin(levels).all():
                raise ValueError(f'Неизвестная категория в {col}')
            for level in levels[1:]:
                cols[f'{col}_{level}'] = (values == level).astype(float)
        return pd.DataFrame(cols, index=frame.index).to_numpy(dtype=float)

    def fit(self, frame):
        self.medians = frame[self.numeric].median()
        self.modes = {c: frame[c].mode().iloc[0] for c in self.categories}
        x = self._encode(frame)
        self.mean, self.std = x.mean(axis=0), x.std(axis=0)
        self.std[self.std == 0] = 1
        self.feature_names = self.numeric + [f'{c}_{v}' for c, levels in self.categories.items() for v in levels[1:]]
        return self

    def transform(self, frame):
        return (self._encode(frame) - self.mean) / self.std


class LogisticRegression:
    def __init__(self, learning_rate=0.1, n_iter=1000, optimizer='gd'):
        if learning_rate <= 0 or n_iter < 1 or optimizer not in ('gd', 'newton'):
            raise ValueError('Проверьте гиперпараметры')
        self.learning_rate, self.n_iter, self.optimizer = learning_rate, n_iter, optimizer

    @staticmethod
    def design(x):
        return np.column_stack([np.ones(len(x)), x])

    def fit(self, x, y):
        x, y = self.design(x), np.asarray(y, dtype=float)
        self.weights = np.zeros(x.shape[1])
        self.loss_history = [log_loss(y, x @ self.weights)]
        for _ in range(self.n_iter):
            p = sigmoid(x @ self.weights)
            grad = x.T @ (p-y) / len(y)
            if self.optimizer == 'gd':
                direction = grad
            else:
                hessian = (x.T * (p*(1-p))) @ x / len(y)
                # Численная стабилизация Гессиана; целевая функция без L2.
                direction = np.linalg.solve(hessian + 1e-8*np.eye(x.shape[1]), grad)
            self.weights -= self.learning_rate * direction
            self.loss_history.append(log_loss(y, x @ self.weights))
        if not np.isfinite(self.weights).all():
            raise FloatingPointError('Модель разошлась')
        return self

    def decision_function(self, x):
        return self.design(x) @ self.weights

    def predict_proba(self, x):
        return sigmoid(self.decision_function(x))

    def predict(self, x):
        return (self.predict_proba(x) >= 0.5).astype(int)
