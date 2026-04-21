from __future__ import annotations

import numpy as np


def _contingency_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    cm = np.zeros((true_labels.size, pred_labels.size), dtype=int)

    true_index = {label: i for i, label in enumerate(true_labels)}
    pred_index = {label: i for i, label in enumerate(pred_labels)}

    for t, p in zip(y_true, y_pred):
        cm[true_index[t], pred_index[p]] += 1
    return cm, true_labels, pred_labels


def _hungarian_min(cost: np.ndarray) -> np.ndarray:
    """Hungarian algorithm for square cost matrix (minimization)."""
    n = cost.shape[0]
    u = np.zeros(n + 1)
    v = np.zeros(n + 1)
    p = np.zeros(n + 1, dtype=int)
    way = np.zeros(n + 1, dtype=int)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = np.inf
            j1 = 0

            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = np.full(n, -1, dtype=int)
    for j in range(1, n + 1):
        if p[j] > 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def best_map_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, np.ndarray]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm, true_labels, pred_labels = _contingency_matrix(y_true, y_pred)
    n = max(cm.shape)
    pad = np.zeros((n, n), dtype=int)
    pad[: cm.shape[0], : cm.shape[1]] = cm

    cost = pad.max() - pad
    assignment = _hungarian_min(cost)

    mapping: dict[int, int] = {}
    for true_i in range(cm.shape[0]):
        pred_i = assignment[true_i]
        if pred_i < cm.shape[1]:
            mapping[int(pred_labels[pred_i])] = int(true_labels[true_i])

    aligned = np.array([mapping.get(int(p), int(p)) for p in y_pred], dtype=int)
    acc = float(np.mean(aligned == y_true))
    return acc, aligned


def normalized_mutual_info(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm, _, _ = _contingency_matrix(y_true, y_pred)
    n = cm.sum()
    if n == 0:
        return 0.0

    pxy = cm / n
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)

    nz = pxy > 0
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])))

    hx = -float(np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = -float(np.sum(py[py > 0] * np.log(py[py > 0])))

    denom = np.sqrt(max(hx * hy, 1e-15))
    return mi / denom


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm, _, _ = _contingency_matrix(np.asarray(y_true), np.asarray(y_pred))
    return float(np.sum(np.max(cm, axis=0)) / np.sum(cm))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1s: list[float] = []

    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall == 0:
            f1s.append(0.0)
        else:
            f1s.append(2.0 * precision * recall / (precision + recall))

    return float(np.mean(f1s)) if f1s else 0.0


def clustering_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    acc, aligned = best_map_accuracy(y_true, y_pred)
    return {
        "acc": acc,
        "nmi": normalized_mutual_info(y_true, y_pred),
        "purity": purity_score(y_true, y_pred),
        "f1": macro_f1(y_true, aligned),
    }
