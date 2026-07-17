"""RBF-SVM classification of consciousness state from network features.

Reproduces the paper's stratified four-fold RBF-SVM procedure with Platt
(sigmoid) probability calibration, a Youden decision threshold for binary
tasks, and a chi-square test on the confusion matrix.

Note (from the paper's own design): hyperparameters are selected with the same
four-fold split used to report performance, so the estimate is not nested and
can be optimistic. Use nested cross-validation for genuinely new claims.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import SubjectFeatures


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    metric: str
    band: str
    density: float | None = None


@dataclass(slots=True)
class ClassificationResult:
    classes: np.ndarray
    predicted: np.ndarray
    probabilities: np.ndarray
    confusion: np.ndarray
    accuracy: float
    chi2: float
    p_value: float
    best_params: dict
    youden_threshold: float | None


def paper_grid_values() -> np.ndarray:
    """Union of the logarithmic C/scale grids stated in the supplement."""
    values = [
        *(2.0 ** np.arange(-5, 6)),
        *(5.0 ** np.arange(-3, 4)),
        *(10.0 ** np.arange(-3, 4)),
    ]
    return np.unique(np.asarray(values, dtype=float))


def build_feature_matrix(
    features: Sequence[SubjectFeatures], specs: Sequence[FeatureSpec]
) -> np.ndarray:
    """Concatenate the requested per-subject feature vectors into a matrix."""
    rows = []
    for subject in features:
        parts = [subject.node_feature(s.metric, s.band, s.density) for s in specs]
        rows.append(np.concatenate(parts))
    return np.asarray(rows)


def svm_cross_validation(
    x: np.ndarray,
    y: Sequence,
    *,
    random_state: int = 17,
    compact_grid: bool = False,
) -> ClassificationResult:
    """Stratified four-fold RBF-SVM with Platt probabilities and Youden cutoff."""
    y = np.asarray(y)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=random_state)
    search_model = Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(kernel="rbf", random_state=random_state)),
    ])
    scales = np.asarray([0.1, 1.0, 10.0]) if compact_grid else paper_grid_values()
    c_values = np.asarray([0.1, 1.0, 10.0]) if compact_grid else paper_grid_values()
    gamma = 1.0 / (2.0 * scales ** 2)
    search = GridSearchCV(
        search_model,
        {"svm__C": c_values, "svm__gamma": gamma},
        scoring="balanced_accuracy", cv=cv, n_jobs=1,
    )
    search.fit(x, y)

    calibrated = CalibratedClassifierCV(
        SVC(kernel="rbf", C=search.best_params_["svm__C"],
            gamma=search.best_params_["svm__gamma"], random_state=random_state),
        method="sigmoid", cv=4,
    )
    prob_model = Pipeline([("scale", StandardScaler()), ("platt", calibrated)])
    probabilities = cross_val_predict(prob_model, x, y, cv=cv,
                                      method="predict_proba", n_jobs=1)

    classes = np.unique(y)
    threshold = None
    if len(classes) == 2:
        yb = (y == classes[1]).astype(int)
        fpr, tpr, thr = roc_curve(yb, probabilities[:, 1])
        threshold = float(thr[np.argmax(tpr - fpr)])
        predicted = np.where(probabilities[:, 1] >= threshold, classes[1], classes[0])
    else:
        predicted = classes[np.argmax(probabilities, axis=1)]

    confusion = confusion_matrix(y, predicted, labels=classes)
    rows = confusion.sum(axis=1) > 0
    cols = confusion.sum(axis=0) > 0
    reduced = confusion[np.ix_(rows, cols)]
    if min(reduced.shape) < 2:
        chi2, p_value = 0.0, 1.0
    else:
        chi2, p_value, _, _ = chi2_contingency(reduced)

    return ClassificationResult(
        classes=classes,
        predicted=predicted,
        probabilities=probabilities,
        confusion=confusion,
        accuracy=float(np.mean(predicted == y)),
        chi2=float(chi2),
        p_value=float(p_value),
        best_params={k: float(v) for k, v in search.best_params_.items()},
        youden_threshold=threshold,
    )
