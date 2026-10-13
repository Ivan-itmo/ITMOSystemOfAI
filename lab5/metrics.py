import numpy as np


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # положительный класс — ядовитый гриб, ошибки считаются относительно класса 1
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp


def accuracy_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, fp, _, tp = confusion_counts(y_true, y_pred)
    denominator = tp + fp
    return tp / denominator if denominator else 0.0


def recall_score_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, _, fn, tp = confusion_counts(y_true, y_pred)
    denominator = tp + fn
    return tp / denominator if denominator else 0.0


def _trapezoid_auc(x: np.ndarray, y: np.ndarray) -> float:
    # суммируем площади трапеций между соседними точками при неубывающем x
    area = 0.0
    for i in range(1, len(x)):
        area += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2.0
    return float(area)


def roc_curve_manual(y_true: np.ndarray, scores: np.ndarray):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    # снижаем порог по уникальным оценкам, крайние пороги дают все ответы 0 и все ответы 1
    thresholds = [float("inf")] + sorted(np.unique(scores).tolist(), reverse=True) + [float("-inf")]

    fpr_values = []
    tpr_values = []

    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_counts(y_true, y_pred)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr_values.append(fpr)
        tpr_values.append(tpr)

    fpr = np.asarray(fpr_values, dtype=float)
    tpr = np.asarray(tpr_values, dtype=float)

    # сохраняем порядок точек с одинаковым fpr, чтобы не исказить вертикальные участки
    order = np.argsort(fpr, kind="stable")
    fpr = fpr[order]
    tpr = tpr[order]
    auc = _trapezoid_auc(fpr, tpr)
    return fpr, tpr, auc


def pr_curve_manual(y_true: np.ndarray, scores: np.ndarray):
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    thresholds = [float("inf")] + sorted(np.unique(scores).tolist(), reverse=True) + [float("-inf")]

    precision_values = []
    recall_values = []

    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        _, fp, fn, tp = confusion_counts(y_true, y_pred)
        # при отсутствии положительных предсказаний задаём начальную точку pr как (0, 1)
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision_values.append(precision)
        recall_values.append(recall)

    recall = np.asarray(recall_values, dtype=float)
    precision = np.asarray(precision_values, dtype=float)
    order = np.argsort(recall, kind="stable")
    recall = recall[order]
    precision = precision[order]
    # площадь под линейно соединёнными точками pr отличается от average precision
    auc = _trapezoid_auc(recall, precision)
    return recall, precision, auc
