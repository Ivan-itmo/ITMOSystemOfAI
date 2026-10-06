"""Воспроизводимый эксперимент: python3 main.py."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from data_processor import (TARGET, SYNTHETIC, load_and_visualize, split_dataframe, add_synthetic_features, prepare_features, plt)
from linear_regression import LinearRegressionManual


def main():
    parser = argparse.ArgumentParser(description='Лабораторная работа: ручная линейная регрессия')
    parser.add_argument('--data', type=Path, default=Path(__file__).with_name('california_housing_train.csv'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results'))
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    df = load_and_visualize(args.data, output)
    clean = df.loc[df[TARGET].notna() & np.isfinite(df[TARGET])].copy()
    train, test = split_dataframe(clean)
    pd.concat([pd.DataFrame({'row_index': train.index, 'split': 'train'}),
               pd.DataFrame({'row_index': test.index, 'split': 'test'})]).to_csv(output / 'split.csv', index=False)

    original = [col for col in df if col != TARGET]
    train, test = add_synthetic_features(train), add_synthetic_features(test)
    specifications = [
        ('Модель 1: исходные признаки', original),
        ('Модель 2: исходные + синтетические', original + SYNTHETIC),
        ('Модель 3: доход и возраст жилья', ['median_income', 'housing_median_age']),
    ]

    results, predictions, coefficients, scales = [], [], [], []
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for number, (name, features) in enumerate(specifications, start=1):
        X_train, X_test, y_train, y_test, processor = prepare_features(train, test, features)
        model = LinearRegressionManual().fit(X_train, y_train)
        prediction = model.predict(X_test)

        results.append({
            'model': name,
            'n_features': X_train.shape[1] - 1,
            'r2_train': model.score(X_train, y_train),
            'r2_test': model.score(X_test, y_test),
            'sse_train': float(np.sum((y_train - model.predict(X_train)) ** 2)),
            'sse_test': float(np.sum((y_test - prediction) ** 2)),
        })

        raw_slopes = model.coefficients[1:] / processor.std.to_numpy()
        raw_intercept = model.coefficients[0] - processor.mean.to_numpy() @ raw_slopes
        for feature, beta, raw_beta in zip(['intercept'] + processor.feature_names,
                                           model.coefficients, np.r_[raw_intercept, raw_slopes]):
            coefficients.append({
                'model': name,
                'feature': feature,
                'coefficient_standardized': beta,
                'coefficient_original_scale': raw_beta,
            })

        for feature in processor.feature_names:
            scales.append({
                'model': name,
                'feature': feature,
                'train_mean': processor.mean[feature],
                'train_std': processor.std[feature],
            })

        predictions.append(pd.DataFrame({
            'model': name,
            'row_index': test.index,
            'actual': y_test,
            'predicted': prediction,
        }))

        ax = axes[number - 1]
        ax.scatter(y_test, prediction, s=5, alpha=.25, color='#4477aa', rasterized=True)
        limits = [min(y_test.min(), prediction.min()), max(y_test.max(), prediction.max())]
        ax.plot(limits, limits, '--', color='#cc3311')
        ax.set_title(f'Model {number}: test R² = {results[-1]["r2_test"]:.4f}')
        ax.set_xlabel('Actual house value')
        ax.set_ylabel('Predicted house value')
        ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0))

    fig.tight_layout()
    fig.savefig(output / 'predictions.png', dpi=150)
    plt.close(fig)

    metrics = pd.DataFrame(results)
    metrics.to_csv(output / 'metrics.csv', index=False)

    coefficient_table = pd.DataFrame(coefficients)
    coefficient_table.to_csv(output / 'coefficients.csv', index=False)

    pd.DataFrame(scales).to_csv(output / 'scaling.csv', index=False)
    pd.concat(predictions).to_csv(output / 'predictions.csv', index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = np.arange(3)
    ax.bar(positions - .18, metrics.r2_train, width=.36, label='Train', color='#4477aa')
    ax.bar(positions + .18, metrics.r2_test, width=.36, label='Test', color='#228833')
    for i, value in enumerate(metrics.r2_test):
        ax.text(i + .18, value + .015, f'{value:.4f}', ha='center')
    ax.set_xticks(positions, ['1: original', '2: + synthetic', '3: income + age'])
    ax.set_ylabel('R²')
    ax.set_title('Comparison on the same train/test split')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'model_comparison.png', dpi=150)
    plt.close(fig)

    print('\n', metrics.to_string(index=False))
    print(f'\nРезультаты и графики: {output.resolve()}')


if __name__ == '__main__':
    main()