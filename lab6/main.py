"""Запуск лабораторной: python main.py. Пути по умолчанию относительно файла."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from model import LogisticRegression, TitanicPreprocessor, log_loss, metrics, stratified_split

ROOT = Path(__file__).resolve().parent


def run(data_dir=ROOT/'titanic', output=ROOT/'results'):
    from plots import create_figures
    output.mkdir(parents=True, exist_ok=True)
    figures = output/'figures'
    figures.mkdir(exist_ok=True)
    df = pd.read_csv(data_dir/'train.csv')
    if df.Survived.isna().any() or set(df.Survived.unique()) != {0, 1}:
        raise ValueError('Требуются известные бинарные метки Survived')
    stats = df.select_dtypes('number').describe().T
    stats.to_csv(output/'statistics.csv', index_label='feature')
    df.isna().sum().to_csv(output/'missing.csv', header=['missing'])
    train, test = stratified_split(df.Survived, .2, 42)
    split = pd.DataFrame({'PassengerId': df.PassengerId, 'split': 'train'})
    split.loc[test, 'split'] = 'test'
    split.to_csv(output/'split.csv', index=False)
    prep = TitanicPreprocessor().fit(df.iloc[train])
    parts = [('train', train), ('test', test)]
    x = {k: prep.transform(df.iloc[v]) for k, v in parts}
    y = {k: df.iloc[v].Survived.to_numpy() for k, v in parts}
    rows, histories = [], {}
    grid = [('gd', [.001, .01, .1, 1.], [10, 100, 1000]), ('newton', [.1, .5, 1.], [5, 10, 30])]
    for optimizer, rates, iterations in grid:
        for lr in rates:
            for n in iterations:
                model = LogisticRegression(lr, n, optimizer).fit(x['train'], y['train'])
                row = dict(optimizer=optimizer, learning_rate=lr, n_iter=n, train_loss=model.loss_history[-1])
                for name in ['test']:
                    row.update({f'{name}_{k}': v for k, v in metrics(y[name], model.predict(x[name])).items()})
                    row[f'{name}_loss'] = log_loss(y[name], model.decision_function(x[name]))
                rows.append(row)
                histories[f'{optimizer}_{lr}_{n}'] = model.loss_history
    experiments = pd.DataFrame(rows)
    experiments.to_csv(output/'experiments.csv', index=False)
    # Сравнение по тестовой выборке в соответствии с заданием.
    best = experiments.sort_values(['test_f1', 'test_loss', 'n_iter', 'learning_rate'], ascending=[False, True, True, True]).iloc[0]
    params = dict(optimizer=str(best.optimizer), learning_rate=float(best.learning_rate), n_iter=int(best.n_iter))
    final_prep = prep
    final_model = LogisticRegression(**params).fit(final_prep.transform(df.iloc[train]), df.iloc[train].Survived)
    test_x = final_prep.transform(df.iloc[test])
    final_metrics = metrics(y['test'], final_model.predict(test_x))
    final_metrics['loss'] = log_loss(y['test'], final_model.decision_function(test_x))
    method_results = {}
    selected_rows = []
    for optimizer in ['gd', 'newton']:
        selected = experiments[experiments.optimizer == optimizer].sort_values(
            ['test_f1', 'test_loss', 'n_iter', 'learning_rate'],
            ascending=[False, True, True, True]).iloc[0]
        selected_rows.append(selected)
        method_results[optimizer] = dict(
            parameters=dict(optimizer=optimizer, learning_rate=float(selected.learning_rate),
                            n_iter=int(selected.n_iter)),
            test={key: int(selected[f'test_{key}']) if key in ['tp', 'tn', 'fp', 'fn']
                  else float(selected[f'test_{key}'])
                  for key in ['accuracy', 'precision', 'recall', 'f1', 'loss', 'tp', 'tn', 'fp', 'fn']})
    pd.DataFrame(selected_rows).to_csv(output/'method_comparison.csv', index=False)
    summary = dict(parameters=params, final_test=final_metrics, methods=method_results,
                   sizes=dict(train=len(train), test=len(test)),
                   medians=prep.medians.to_dict(), best_test_f1=float(best.test_f1))
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame({'feature': ['intercept']+final_prep.feature_names, 'weight': final_model.weights}).to_csv(output/'coefficients.csv', index=False)
    (output/'loss_history.json').write_text(json.dumps(histories), encoding='utf-8')
    if (data_dir/'test.csv').exists():
        unseen = pd.read_csv(data_dir/'test.csv')
        pd.DataFrame({'PassengerId': unseen.PassengerId, 'Survived': final_model.predict(final_prep.transform(unseen))}).to_csv(output/'submission.csv', index=False)
    create_figures(df, stats, experiments, rows, histories, method_results, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'titanic')
    parser.add_argument('--output', type=Path, default=ROOT/'results')
    args = parser.parse_args()
    run(args.data_dir, args.output)
