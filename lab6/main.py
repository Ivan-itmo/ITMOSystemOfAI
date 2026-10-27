"""Запуск лабораторной: python main.py. Пути по умолчанию относительно файла."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from model import LogisticRegression, TitanicPreprocessor, log_loss, metrics, stratified_split

ROOT = Path(__file__).resolve().parent


def run(data_dir=ROOT/'titanic', output=ROOT/'results'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    figures = output/'figures'
    figures.mkdir(exist_ok=True)
    df = pd.read_csv(data_dir/'train.csv')
    if df.Survived.isna().any() or set(df.Survived.unique()) != {0, 1}:
        raise ValueError('Требуются известные бинарные метки Survived')
    stats = df.select_dtypes('number').describe().T
    stats.to_csv(output/'statistics.csv', index_label='feature')
    df.isna().sum().to_csv(output/'missing.csv', header=['missing'])
    development, test = stratified_split(df.Survived, .2, 42)
    tr, va = stratified_split(df.iloc[development].Survived, .25, 43)
    train, val = development[tr], development[va]
    split = pd.DataFrame({'PassengerId': df.PassengerId, 'split': 'train'})
    split.loc[val, 'split'] = 'validation'
    split.loc[test, 'split'] = 'test'
    split.to_csv(output/'split.csv', index=False)
    prep = TitanicPreprocessor().fit(df.iloc[train])
    parts = [('train', train), ('val', val), ('test', test)]
    x = {k: prep.transform(df.iloc[v]) for k, v in parts}
    y = {k: df.iloc[v].Survived.to_numpy() for k, v in parts}
    rows, histories = [], {}
    grid = [('gd', [.001, .01, .1, 1.], [10, 100, 1000]), ('newton', [.1, .5, 1.], [5, 10, 30])]
    for optimizer, rates, iterations in grid:
        for lr in rates:
            for n in iterations:
                model = LogisticRegression(lr, n, optimizer).fit(x['train'], y['train'])
                row = dict(optimizer=optimizer, learning_rate=lr, n_iter=n, train_loss=model.loss_history[-1])
                for name in ['val', 'test']:
                    row.update({f'{name}_{k}': v for k, v in metrics(y[name], model.predict(x[name])).items()})
                    row[f'{name}_loss'] = log_loss(y[name], model.decision_function(x[name]))
                rows.append(row)
                histories[f'{optimizer}_{lr}_{n}'] = model.loss_history
    experiments = pd.DataFrame(rows)
    experiments.to_csv(output/'experiments.csv', index=False)
    # Заранее фиксированное правило; тест не участвует в выборе параметров.
    best = experiments.sort_values(['val_f1', 'val_loss', 'n_iter', 'learning_rate'], ascending=[False, True, True, True]).iloc[0]
    params = dict(optimizer=str(best.optimizer), learning_rate=float(best.learning_rate), n_iter=int(best.n_iter))
    final_prep = TitanicPreprocessor().fit(df.iloc[development])
    final_model = LogisticRegression(**params).fit(final_prep.transform(df.iloc[development]), df.iloc[development].Survived)
    test_x = final_prep.transform(df.iloc[test])
    final_metrics = metrics(y['test'], final_model.predict(test_x))
    final_metrics['loss'] = log_loss(y['test'], final_model.decision_function(test_x))
    baseline = metrics(y['test'], np.full(len(test), int(df.iloc[development].Survived.mean() >= .5)))
    summary = dict(parameters=params, final_test=final_metrics, baseline=baseline,
                   sizes=dict(train=len(train), validation=len(val), test=len(test), development=len(development)),
                   medians=prep.medians.to_dict(), best_validation_f1=float(best.val_f1))
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame({'feature': ['intercept']+final_prep.feature_names, 'weight': final_model.weights}).to_csv(output/'coefficients.csv', index=False)
    (output/'loss_history.json').write_text(json.dumps(histories), encoding='utf-8')
    if (data_dir/'test.csv').exists():
        unseen = pd.read_csv(data_dir/'test.csv')
        pd.DataFrame({'PassengerId': unseen.PassengerId, 'Survived': final_model.predict(final_prep.transform(unseen))}).to_csv(output/'submission.csv', index=False)
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 140})
    def save(name):
        plt.tight_layout(); plt.savefig(figures/name, bbox_inches='tight'); plt.close()
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    df.Survived.value_counts().sort_index().plot.bar(ax=axs[0, 0], title='Survived: counts', rot=0)
    df.isna().sum().plot.bar(ax=axs[0, 1], title='Missing values', rot=45)
    for target, group in df.groupby('Survived'):
        axs[1, 0].hist(group.Age.dropna(), bins=np.arange(0, 91, 10), alpha=.5, label=str(target))
    axs[1, 0].set_title('Age by Survived'); axs[1, 0].legend()
    df.groupby(['Sex', 'Pclass']).Survived.mean().unstack().plot.bar(ax=axs[1, 1], title='Survival by sex and class', rot=0)
    save('dataset.png')
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    for col, ax in zip(['Age', 'SibSp', 'Parch', 'Fare'], axs.flat):
        df.boxplot(column=col, ax=ax); ax.set_title(col)
    save('boxplots.png')
    table = stats.drop(index='PassengerId')
    normalized = table/table.abs().max().replace(0, 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(normalized, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table)), table.index)
    for i in range(len(table)):
        for j in range(len(table.columns)):
            ax.text(j, i, f'{table.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8, color='white' if normalized.iloc[i, j] > .6 else 'black')
    ax.set_title('Descriptive statistics (color scaled per column)'); fig.colorbar(im, ax=ax)
    save('statistics.png')
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for optimizer, ax in zip(['gd', 'newton'], axs):
        for lr, group in experiments[experiments.optimizer == optimizer].groupby('learning_rate'):
            ax.plot(group.n_iter, group.val_f1, marker='o', label=f'lr={lr}')
        ax.set_xscale('log'); ax.set_xlabel('Iterations'); ax.set_ylabel('Validation F1')
        ax.set_title(optimizer); ax.legend(); ax.grid(alpha=.25)
    save('hyperparameters.png')
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for optimizer, n, ax in [('gd', 1000, axs[0]), ('newton', 30, axs[1])]:
        for row in rows:
            if row['optimizer'] == optimizer and row['n_iter'] == n:
                history = histories[f"{optimizer}_{row['learning_rate']}_{n}"]
                ax.plot(range(len(history)), history, label=f"lr={row['learning_rate']}")
        ax.set_title(f'{optimizer}: training log loss'); ax.set_xlabel('Iterations'); ax.legend()
    save('loss.png')
    cm = np.array([[final_metrics['tn'], final_metrics['fp']], [final_metrics['fn'], final_metrics['tp']]])
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title('Final test confusion matrix'); save('confusion_matrix.png')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'titanic')
    parser.add_argument('--output', type=Path, default=ROOT/'results')
    args = parser.parse_args()
    run(args.data_dir, args.output)
