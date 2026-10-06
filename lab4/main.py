from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_utils import (
    encode_categorical_features,
    impute_numeric,
    load_dataset,
    standardize,
    stratified_split,
)
from knn import KNNClassifier, accuracy_score, confusion_matrix


TARGET = "Wine"
SEED = 42
TEST_SIZE = 0.25
K_VALUES = [1, 3, 5, 7, 10]
RANDOM_FEATURE_COUNT = 5
# Заранее выбранные признаки второй модели.
FIXED_FEATURES = [
    "Alcohol",
    "Flavanoids",
    "Color intensity",
    "OD280/OD315 of diluted wines",
    "Proline",
]
FEATURES_3D = ["Alcohol", "Flavanoids", "Color intensity"]


def save_statistics(df, output_dir):
    numeric = df.select_dtypes(include=[np.number])
    # Количество, среднее, стандартное отклонение, экстремумы и квартили.
    statistics = numeric.describe().T
    statistics.to_csv(output_dir / "statistics.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis("off")
    shown = statistics.round(3).reset_index().rename(columns={"index": "Feature"})
    table = ax.table(
        cellText=shown.values,
        colLabels=shown.columns,
        cellLoc="center",
        loc="center",
        colWidths=[0.20] + [0.10] * (len(shown.columns) - 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.3)
    ax.set_title("Dataset statistics")
    fig.tight_layout()
    fig.savefig(output_dir / "statistics_table.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    feature_columns = [column for column in numeric.columns if column != TARGET]
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(15, 11))
    axes = axes.flatten()
    for ax, column in zip(axes, feature_columns):
        ax.hist(numeric[column].dropna(), bins=18)
        ax.set_title(column, fontsize=9)
        ax.grid(alpha=0.2)
    for ax in axes[len(feature_columns):]:
        ax.axis("off")
    fig.suptitle("Feature distributions")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_distributions.png", dpi=180)
    plt.close(fig)


def save_3d_plot(df, output_dir):
    # Три признака задают координаты, класс вина — цвет точки.
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        df[FEATURES_3D[0]],
        df[FEATURES_3D[1]],
        df[FEATURES_3D[2]],
        c=df[TARGET],
        s=35,
    )
    ax.set_xlabel(FEATURES_3D[0])
    ax.set_ylabel(FEATURES_3D[1])
    ax.set_zlabel(FEATURES_3D[2])
    ax.set_title("3D feature visualization")
    legend = ax.legend(*scatter.legend_elements(), title="Wine class")
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(output_dir / "features_3d.png", dpi=180)
    plt.close(fig)


def evaluate_model(name, features, X_train, X_test, y_train, y_test, labels, output_dir):
    train_part = X_train[features]
    test_part = X_test[features]
    train_part, test_part = impute_numeric(train_part, test_part)
    train_scaled, test_scaled = standardize(train_part, test_part)

    rows = []
    matrices = {}

    # Сравниваем значения k на одной и той же тестовой выборке.
    for k in K_VALUES:
        model = KNNClassifier(k=k)
        model.fit(train_scaled.to_numpy(), y_train)
        predictions = model.predict(test_scaled.to_numpy())
        accuracy = accuracy_score(y_test, predictions)
        matrix = confusion_matrix(y_test, predictions, labels)
        rows.append({"model": name, "k": k, "accuracy": accuracy})
        matrices[k] = matrix

    fig, axes = plt.subplots(1, len(K_VALUES), figsize=(18, 4))
    axes = np.atleast_1d(axes).flatten()
    for ax, k in zip(axes, K_VALUES):
        matrix = matrices[k]
        image = ax.imshow(matrix)
        ax.set_title(f"k = {k}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.suptitle(f"Confusion matrices: {name}")
    fig.subplots_adjust(top=0.80, wspace=0.35)
    fig.savefig(output_dir / f"confusion_matrices_{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    return rows


def save_accuracy_plot(results, output_dir):
    results_df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, group in results_df.groupby("model"):
        group = group.sort_values("k")
        ax.plot(group["k"], group["accuracy"], marker="o", label=name)
    ax.set_xlabel("k")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(K_VALUES)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("k-NN accuracy")
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_by_k.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="WineDataset.csv")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    # Объекты без известного класса исключаем из обучения и оценки.
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype(int)

    print(f"Количество строк: {len(df)}")
    print(f"Количество столбцов: {len(df.columns)}")
    print("Пропущенные значения:")
    print(df.isna().sum().to_string())

    save_statistics(df, output_dir)
    save_3d_plot(df, output_dir)

    X, categorical = encode_categorical_features(df, TARGET)
    y = df[TARGET].to_numpy()
    labels = np.sort(np.unique(y))

    print("Категориальные признаки:", ", ".join(categorical) if categorical else "нет")

    train_idx, test_idx = stratified_split(y, test_size=TEST_SIZE, seed=SEED)
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    available_original = [column for column in df.columns if column != TARGET]
    # Фиксированный seed делает случайный набор признаков воспроизводимым.
    rng = np.random.default_rng(SEED)
    random_features = rng.choice(
        available_original,
        size=min(RANDOM_FEATURE_COUNT, len(available_original)),
        replace=False,
    ).tolist()

    missing_fixed = [feature for feature in FIXED_FEATURES if feature not in X.columns]
    if missing_fixed:
        raise ValueError(f"Отсутствуют фиксированные признаки: {missing_fixed}")

    print("Модель 1 — случайные признаки:", ", ".join(random_features))
    print("Модель 2 — фиксированные признаки:", ", ".join(FIXED_FEATURES))
    print(f"Размер обучающей выборки: {len(train_idx)}")
    print(f"Размер тестовой выборки: {len(test_idx)}")

    results = []
    results.extend(
        evaluate_model(
            "model_1_random",
            random_features,
            X_train,
            X_test,
            y_train,
            y_test,
            labels,
            output_dir,
        )
    )
    results.extend(
        evaluate_model(
            "model_2_fixed",
            FIXED_FEATURES,
            X_train,
            X_test,
            y_train,
            y_test,
            labels,
            output_dir,
        )
    )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "evaluation.csv", index=False, encoding="utf-8-sig")
    save_accuracy_plot(results, output_dir)

    print("\nТочность классификации:")
    print(results_df.to_string(index=False, formatters={"accuracy": "{:.4f}".format}))

    best = results_df.sort_values(["accuracy", "k"], ascending=[False, True]).iloc[0]
    print(
        f"\nЛучший результат: {best['model']}, k={int(best['k'])}, "
        f"accuracy={best['accuracy']:.4f}"
    )
    print(f"Результаты сохранены в: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
