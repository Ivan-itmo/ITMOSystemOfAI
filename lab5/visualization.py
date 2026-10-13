from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_class_distribution(y: np.ndarray, output_path: str | Path):
    values, counts = np.unique(y, return_counts=True)
    labels = ["Съедобные" if value == 0 else "Ядовитые" for value in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, counts)
    ax.set_title("Распределение классов")
    ax.set_ylabel("Количество")
    for i, count in enumerate(counts):
        ax.text(i, count, str(int(count)), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_feature_cardinality(df: pd.DataFrame, output_path: str | Path):
    counts = df.nunique().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(counts.index, counts.values)
    ax.set_title("Количество уникальных значений признаков")
    ax.set_ylabel("Уникальных категорий")
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_roc_pr(fpr, tpr, roc_auc, recall, precision, pr_auc, output_path: str | Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, marker="o", markersize=3, label=f"AUC = {roc_auc:.4f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", label="Случайный классификатор")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC-кривая")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(recall, precision, marker="o", markersize=3, label=f"AUC = {pr_auc:.4f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall кривая")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(tn: int, fp: int, fn: int, tp: int, output_path: str | Path):
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix)
    ax.set_xticks([0, 1], ["Съедобный", "Ядовитый"])
    ax.set_yticks([0, 1], ["Съедобный", "Ядовитый"])
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title("Матрица ошибок")

    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
