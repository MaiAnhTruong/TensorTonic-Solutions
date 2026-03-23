def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.

    For single-label multi-class classification, micro-F1 is computed by
    aggregating TP, FP, and FN across all classes first.

    Returns:
        float in [0, 1]
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if not y_true:
        return 0.0

    tp = fp = fn = 0

    labels = set(y_true) | set(y_pred)

    for label in labels:
        for yt, yp in zip(y_true, y_pred):
            if yt == label and yp == label:
                tp += 1
            elif yt != label and yp == label:
                fp += 1
            elif yt == label and yp != label:
                fn += 1

    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom