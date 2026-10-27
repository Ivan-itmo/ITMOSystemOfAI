from pathlib import Path

import numpy as np

from data_utils import (
    CategoricalPreprocessor,
    load_dataset,
    select_random_sqrt_features,
    stratified_train_test_split,
)
from decision_tree import DecisionTreeClassifierManual
from metrics import (
    accuracy_score_manual,
    confusion_counts,
    pr_curve_manual,
    precision_score_manual,
    recall_score_manual,
    roc_curve_manual,
)
from visualization import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_cardinality,
    plot_roc_pr,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "agaricus-lepiota.data"
RESULTS_DIR = BASE_DIR / "results"
TARGET = "class"
TEST_SIZE = 0.2
SPLIT_SEED = 67
FEATURE_SEED = 10
FEATURE_MODE = "preselected" # "random" для случайного выбора
PRESELECTED_FEATURES = [
    "odor", # запах - 0.391017
    "gill-size", # размер пластинок - 0.249915
    "stalk-surface-above-ring", # поверхность ножки над кольцом - 0.249915
    "spore-print-color", # цвет спорового отпечатка - 0.220998

    # выбраны по лучшему gain ratio (после перевода стоит)
]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    df = load_dataset(DATA_PATH)
    feature_names = [column for column in df.columns if column != TARGET]
    if FEATURE_MODE == "random":
        selected_features = select_random_sqrt_features(feature_names, seed=FEATURE_SEED)
    elif FEATURE_MODE == "preselected":
        selected_features = PRESELECTED_FEATURES.copy()
        if len(selected_features) != int(np.sqrt(len(feature_names))):
            raise ValueError("Нужно заранее выбрать int(sqrt(n)) признаков")
        if len(set(selected_features)) != len(selected_features) or not set(selected_features) <= set(feature_names):
            raise ValueError("Выбранные признаки должны существовать и не повторяться")
    else:
        raise ValueError("FEATURE_MODE должен быть random или preselected")

    train_df, test_df = stratified_train_test_split(df, TARGET, TEST_SIZE, SPLIT_SEED)

    # вычисляем замены пропусков только на обучающей выборке, чтобы избежать утечки данных
    preprocessor = CategoricalPreprocessor().fit(train_df[selected_features])
    X_train = preprocessor.transform(train_df[selected_features])
    X_test = preprocessor.transform(test_df[selected_features])

    # e -> 0 (съедобный), p -> 1 (ядовитый)
    y_train = (train_df[TARGET].to_numpy() == "p").astype(int)
    y_test = (test_df[TARGET].to_numpy() == "p").astype(int)

    tree = DecisionTreeClassifierManual(min_samples_split=2).fit(X_train, y_train)
    predictions = tree.predict(X_test)
    probabilities = tree.predict_proba(X_test)

    accuracy = accuracy_score_manual(y_test, predictions)
    precision = precision_score_manual(y_test, predictions)
    recall = recall_score_manual(y_test, predictions)
    tn, fp, fn, tp = confusion_counts(y_test, predictions)

    fpr, tpr, roc_auc = roc_curve_manual(y_test, probabilities)
    pr_recall, pr_precision, pr_auc = pr_curve_manual(y_test, probabilities)

    plot_class_distribution((df[TARGET].to_numpy() == "p").astype(int), RESULTS_DIR / "class_distribution.png")
    plot_feature_cardinality(df[feature_names], RESULTS_DIR / "feature_cardinality.png")
    plot_confusion_matrix(tn, fp, fn, tp, RESULTS_DIR / "confusion_matrix.png")
    plot_roc_pr(fpr, tpr, roc_auc, pr_recall, pr_precision, pr_auc, RESULTS_DIR / "roc_pr_curves.png")

    (RESULTS_DIR / "tree.txt").write_text(tree.export_text(), encoding="utf-8")

    summary = (
        f"Строк после удаления дубликатов: {len(df)}\n"
        f"Признаков всего: {len(feature_names)}\n"
        f"sqrt(n) -> выбрано признаков: {len(selected_features)}\n"
        f"Режим выбора признаков: {FEATURE_MODE}\n"
        f"Выбранные признаки: {', '.join(selected_features)}\n"
        f"Обучающая выборка: {len(train_df)}\n"
        f"Тестовая выборка: {len(test_df)}\n"
        f"Accuracy: {accuracy:.6f}\n"
        f"Precision: {precision:.6f}\n"
        f"Recall: {recall:.6f}\n"
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}\n"
        f"AUC-ROC: {roc_auc:.6f}\n"
        f"AUC-PR: {pr_auc:.6f}\n"
    )

    (RESULTS_DIR / "metrics.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
