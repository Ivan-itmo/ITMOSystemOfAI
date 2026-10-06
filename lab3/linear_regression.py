import numpy as np


def solve_manual(matrix, rhs):
    """Метод Гаусса с выбором ведущего элемента; свободные переменные = 0.

    При вырожденности возвращает одно решение, не обязательно минимальной нормы.
    """
    a = np.column_stack((matrix, rhs)).astype(float)
    n = len(rhs)
    tolerance = 1e-12 * max(1.0, np.max(np.abs(matrix)))
    pivots, row = [], 0
    for col in range(n):
        pivot = row + int(np.argmax(np.abs(a[row:, col])))
        if abs(a[pivot, col]) <= tolerance:
            continue
        a[[row, pivot]] = a[[pivot, row]]
        for i in range(row + 1, n):
            factor = a[i, col] / a[row, col]
            a[i, col:] -= factor * a[row, col:]
        pivots.append((row, col))
        row += 1
        if row == n:
            break
    solution = np.zeros(n)
    for row, col in reversed(pivots):
        solution[col] = (a[row, -1] - a[row, col + 1:n] @ solution[col + 1:]) / a[row, col]
    return solution


class LinearRegressionManual:
    """Минимизирует SSE = sum((y - X @ beta)**2).

    Нулевой градиент: (X.T @ X) beta = X.T @ y.
    X должен включать столбец единиц для свободного члена.
    """
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not len(y):
            raise ValueError('Ожидаются непустые X[n, p] и y[n].')
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError('X и y должны содержать только конечные числа.')
        self.coefficients = solve_manual(X.T @ X, X.T @ y)
        return self

    def predict(self, X):
        if self.coefficients is None:
            raise RuntimeError('Сначала вызовите fit().')
        return np.asarray(X, dtype=float) @ self.coefficients

    def score(self, X, y):
        y = np.asarray(y, dtype=float)
        residual = np.sum((y - self.predict(X)) ** 2)
        total = np.sum((y - y.mean()) ** 2)
        return float(1 - residual / total) if total > 0 else float('nan')
