from data_utils import CategoricalPreprocessor, load_dataset, stratified_train_test_split
from decision_tree import DecisionTreeClassifierManual
from main import DATA_PATH, RESULTS_DIR, SPLIT_SEED, TARGET, TEST_SIZE


def main():
    train_df, _ = stratified_train_test_split(
        load_dataset(DATA_PATH), TARGET, TEST_SIZE, SPLIT_SEED
    )
    # тестовая выборка не участвует в обработке пропусков и оценке признаков
    features = train_df.drop(columns=TARGET)
    features = CategoricalPreprocessor().fit(features).transform(features)
    labels = (train_df[TARGET].to_numpy() == "p").astype(int)
    tree = DecisionTreeClassifierManual()
    ranking = sorted(
        [(name, tree._gain_ratio(features[name], labels)) for name in features],
        key=lambda item: item[1],
        reverse=True,
    )
    report = (
        f"Ранжирование по Gain Ratio на обучающей выборке\n"
        f"SPLIT_SEED={SPLIT_SEED}, TEST_SIZE={TEST_SIZE}\n"
        + "\n".join(f"{name}: {score:.6f}" for name, score in ranking)
        + "\n"
    )
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "feature_ranking.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
