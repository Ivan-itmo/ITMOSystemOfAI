import numpy as np


class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        if self.k < 1 or self.k > len(self.X_train):
            raise ValueError("k must be between 1 and the number of training samples")
        return self

    def _predict_one(self, x):
        # Выбираем k соседей с наименьшим евклидовым расстоянием.
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        indices = np.argsort(distances)[: self.k]
        labels = self.y_train[indices]
        # Класс определяем большинством голосов соседей.
        unique, counts = np.unique(labels, return_counts=True)
        max_count = counts.max()
        candidates = unique[counts == max_count]

        if len(candidates) == 1:
            return candidates[0]

        # При равенстве голосов сравниваем средние расстояния до соседей класса.
        best_label = candidates[0]
        best_distance = np.inf
        for label in candidates:
            mean_distance = distances[indices][labels == label].mean()
            if mean_distance < best_distance:
                best_distance = mean_distance
                best_label = label
        return best_label

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true, y_pred, labels):
    labels = np.asarray(labels)
    index = {label: i for i, label in enumerate(labels)}
    # Строки — истинные классы, столбцы — предсказанные.
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[index[true], index[pred]] += 1
    return matrix
