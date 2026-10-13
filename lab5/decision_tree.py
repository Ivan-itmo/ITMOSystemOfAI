from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class TreeNode:
    prediction: int
    positive_probability: float
    feature: Optional[str] = None
    # каждому значению признака соответствует отдельная ветвь небинарного дерева
    children: Dict[str, "TreeNode"] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class DecisionTreeClassifierManual:
    def __init__(self, min_samples_split: int = 2, max_depth: Optional[int] = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root: Optional[TreeNode] = None

    @staticmethod
    def _entropy(y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return float(-np.sum(probabilities * np.log2(probabilities)))

    def _information_gain(self, x: pd.Series, y: np.ndarray) -> float:
        parent_entropy = self._entropy(y)
        weighted_entropy = 0.0

        values = x.astype(str).to_numpy()
        for value in np.unique(values):
            mask = values == value
            weighted_entropy += (mask.sum() / len(y)) * self._entropy(y[mask])

        # выигрыш равен уменьшению энтропии с учётом размеров дочерних групп
        return parent_entropy - weighted_entropy

    def _split_info(self, x: pd.Series) -> float:
        return self._entropy(x.astype(str).to_numpy())

    def _gain_ratio(self, x: pd.Series, y: np.ndarray) -> float:
        split_info = self._split_info(x)
        if split_info == 0.0:
            return 0.0
        return self._information_gain(x, y) / split_info

    @staticmethod
    def _majority_class(y: np.ndarray) -> int:
        values, counts = np.unique(y, return_counts=True)
        return int(values[np.argmax(counts)])

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        X = X.reset_index(drop=True).astype(str)
        y = np.asarray(y, dtype=int)
        self.root = self._build_tree(X, y, list(X.columns), depth=0)
        return self

    def _build_tree(self, X: pd.DataFrame, y: np.ndarray, features: list[str], depth: int) -> TreeNode:
        prediction = self._majority_class(y)
        # доля класса 1 в узле служит оценкой вероятности для roc и pr
        positive_probability = float(np.mean(y))
        node = TreeNode(prediction=prediction, positive_probability=positive_probability)

        if len(np.unique(y)) == 1:
            return node
        if not features or len(y) < self.min_samples_split:
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            return node

        gain_ratios = [(feature, self._gain_ratio(X[feature], y)) for feature in features]
        best_feature, best_gain_ratio = max(gain_ratios, key=lambda item: item[1])

        if best_gain_ratio <= 1e-12:
            return node

        node.feature = best_feature
        remaining = [feature for feature in features if feature != best_feature]

        values = X[best_feature].astype(str).to_numpy()
        for value in np.unique(values):
            mask = values == value
            child_X = X.loc[mask, remaining].reset_index(drop=True)
            child_y = y[mask]
            node.children[value] = self._build_tree(child_X, child_y, remaining, depth + 1)

        return node

    def _traverse(self, row: pd.Series) -> TreeNode:
        if self.root is None:
            raise RuntimeError("Дерево не обучено")

        node = self.root
        while not node.is_leaf:
            value = str(row[node.feature])
            child = node.children.get(value)
            if child is None:
                # для неизвестной категории используем предсказание текущего узла
                return node
            node = child
        return node

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.astype(str)
        return np.array([self._traverse(row).prediction for _, row in X.iterrows()], dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.astype(str)
        return np.array([self._traverse(row).positive_probability for _, row in X.iterrows()], dtype=float)

    def export_text(self) -> str:
        if self.root is None:
            return ""

        lines: list[str] = []

        def walk(node: TreeNode, prefix: str, condition: str):
            label = f"класс={node.prediction}, P(ядовитый)={node.positive_probability:.3f}"
            if node.is_leaf:
                lines.append(f"{prefix}{condition} -> {label}")
                return

            lines.append(f"{prefix}{condition} -> признак: {node.feature} ({label})")
            for value, child in sorted(node.children.items()):
                walk(child, prefix + "  ", f"{node.feature}={value}")

        walk(self.root, "", "корень")
        return "\n".join(lines)
