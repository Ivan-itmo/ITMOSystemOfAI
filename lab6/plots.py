"""Русскоязычные графики и корреляционный анализ исходных данных."""
import numpy as np

LABELS = {'PassengerId': 'Номер пассажира', 'Survived': 'Выживание',
          'Pclass': 'Класс билета', 'Age': 'Возраст', 'SibSp': 'Братья, сёстры, супруги',
          'Parch': 'Родители и дети', 'Fare': 'Стоимость билета', 'Sex': 'Пол',
          'Name': 'Имя', 'Ticket': 'Билет', 'Cabin': 'Каюта', 'Embarked': 'Порт посадки',
          'Female': 'Женщина'}
METHODS = {'gd': 'Градиентный спуск', 'newton': 'Метод Ньютона'}


def create_figures(df, stats, experiments, rows, histories, method_results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 140, 'font.family': 'DejaVu Sans'})

    def save(name):
        plt.tight_layout()
        plt.savefig(figures/name, bbox_inches='tight')
        plt.close()

    fig, axs = plt.subplots(2, 2, figsize=(11, 7))
    for col, ax in zip(['Age', 'SibSp', 'Parch', 'Fare'], axs.flat):
        df[[col]].rename(columns=LABELS).boxplot(ax=ax)
        ax.set_title(LABELS[col])
        ax.set_ylabel('Лет' if col == 'Age' else 'Значение признака')
    save('boxplots.png')
    table = stats.drop(index='PassengerId')
    normalized = table/table.abs().max().replace(0, 1)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(normalized, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(table.columns)), ['Количество', 'Среднее', 'Станд. откл.', 'Минимум', 'Квантиль 25%', 'Медиана', 'Квантиль 75%', 'Максимум'], rotation=30, ha='right')
    ax.set_yticks(range(len(table)), [LABELS[c] for c in table.index])
    for i in range(len(table)):
        for j in range(len(table.columns)):
            ax.text(j, i, f'{table.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if normalized.iloc[i, j] > .6 else 'black')
    ax.set_title('Описательная статистика до заполнения пропусков')
    fig.colorbar(im, ax=ax, label='Относительное значение в столбце')
    save('statistics.png')

    # Анализ без заполнения пропусков: каждая пара использует совместно известные значения.
    corr_data = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].copy()
    corr_data['Female'] = df.Sex.map({'female': 1., 'male': 0.})
    known = corr_data.notna().astype(int)
    (known.T @ known).rename(index=LABELS, columns=LABELS).to_csv(output/'correlation_counts.csv', index_label='Признак')
    corr = corr_data.corr(method='pearson')
    corr = corr.rename(index=LABELS, columns=LABELS)
    corr.to_csv(output/'correlation_pearson.csv', index_label='Признак')
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=40, ha='right')
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.iloc[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='white' if abs(val) > .6 else 'black')
    ax.set_title('Матрица корреляций Пирсона')
    fig.colorbar(im, ax=ax, label='Коэффициент корреляции')
    save('correlation_pearson.png')

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for optimizer, ax in zip(['gd', 'newton'], axs):
        for lr, group in experiments[experiments.optimizer == optimizer].groupby('learning_rate'):
            ax.plot(group.n_iter, group.test_f1, marker='o', label=f'Шаг α = {lr}')
        ax.set_xscale('log')
        ax.set(xlabel='Число итераций', ylabel='F1 на тестовой выборке', title=METHODS[optimizer])
        ax.legend(); ax.grid(alpha=.25)
    save('hyperparameters.png')
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    for optimizer, n, ax in [('gd', 1000, axs[0]), ('newton', 30, axs[1])]:
        for row in rows:
            if row['optimizer'] == optimizer and row['n_iter'] == n:
                history = histories[f"{optimizer}_{row['learning_rate']}_{n}"]
                ax.plot(range(len(history)), history, label=f"Шаг α = {row['learning_rate']}")
        ax.set(title=METHODS[optimizer], xlabel='Число итераций', ylabel='Кросс-энтропия на обучении')
        ax.legend()
    save('loss.png')
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for optimizer, ax in zip(['gd', 'newton'], axs):
        result = method_results[optimizer]
        m, params = result['test'], result['parameters']
        cm = np.array([[m['tn'], m['fp']], [m['fn'], m['tp']]])
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=110)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > 55 else 'black')
        ax.set_xticks([0, 1], ['Погиб', 'Выжил'])
        ax.set_yticks([0, 1], ['Погиб', 'Выжил'])
        ax.set(xlabel='Предсказанный класс', ylabel='Истинный класс',
               title=f"{METHODS[optimizer]}\nα = {params['learning_rate']}, итераций: {params['n_iter']}")
    fig.suptitle('Матрицы ошибок лучших вариантов двух методов на тестовой выборке')
    save('confusion_matrix.png')
