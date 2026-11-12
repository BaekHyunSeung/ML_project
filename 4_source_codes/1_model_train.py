# 모델별 파라미터 탐색과 최종 모델 저장
# 탐색 중에는 파일을 만들지 않는다.
# 검증이 끝나면 파라미터와 MAE, RMSE, R² 표를 출력하고,
# 가장 좋은 파라미터로 학습 데이터만 다시 학습해 저장한다.

import importlib
import sys
from pathlib import Path

import joblib
import pandas as pd

common = importlib.import_module("0_common")

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "3_models_train_codes"
SAVED_MODELS_DIR = PROJECT_DIR / "5_models"
BAYESIAN_NAME = "bayesian_linear_regression"
GAUSSIAN_NAME = "gaussian_process"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

tree = importlib.import_module("1_decision_tree_regression")
bayesian = importlib.import_module("0_bayesian_regression")
gaussian = importlib.import_module("2_gaussian")

TREE_COLUMNS = [
    "total_area",
    "cooling_area",
    "solar_capacity",
    "ess_capacity",
    "pcs_capacity",
    "temperature",
    "humidity",
    "wind_speed",
    "month",
    "hour",
    "weekday",
]
METRIC_COLUMNS = ("MAE", "RMSE", "R2")


def summarize(rows):
    """파라미터별 검증 점수를 출력하고, MAE가 가장 작은 파라미터를 반환한다."""
    table = pd.DataFrame(rows)
    print(
        table.to_string(
            index=False,
            formatters={
                "MAE": "{:.4f}".format,
                "RMSE": "{:.4f}".format,
                "R2": "{:.4f}".format,
            },
        )
    )
    best_index = table.sort_values(["MAE", "RMSE"], ascending=True).index[0]
    best_params = {
        key: value
        for key, value in rows[best_index].items()
        if key not in METRIC_COLUMNS
    }
    return best_params, table


def _append_score(rows, params, y_true, y_pred):
    rows.append({**params, **common.regression_metrics(y_true, y_pred)})


def search_tree(train, valid, candidates):
    """결정트리 후보 파라미터를 검증 점수로 비교한다. 파일은 만들지 않는다."""
    if not candidates:
        raise ValueError("candidates가 비어 있습니다.")
    x_train = train[TREE_COLUMNS]
    y_train = train[common.TARGET]
    x_valid = valid[TREE_COLUMNS]
    y_valid = valid[common.TARGET]
    rows = []
    for params in candidates:
        model = tree.train_tree(x_train, y_train, **params)
        _append_score(rows, params, y_valid, model.predict(x_valid))
    return summarize(rows)


def search_bayesian(train, valid, candidates):
    """베이지안 후보 파라미터를 검증 점수로 비교한다. 파일은 만들지 않는다."""
    if not candidates:
        raise ValueError("candidates가 비어 있습니다.")
    y_train = train[common.TARGET]
    y_valid = valid[common.TARGET]
    rows = []
    for params in candidates:
        model = bayesian.train_bayesian(train, y_train, **params)
        _append_score(rows, params, y_valid, model.predict(valid))
    return summarize(rows)


def search_gaussian(train, valid, candidates, rows_per_building=gaussian.ROWS_PER_BUILDING):
    """가우시안 후보 파라미터를 유형 하나의 모델로 검증 점수와 비교한다.

    학습은 건물마다 rows_per_building행만 뽑아서 한다. 검증은 검증 구간 전체로 한다.
    파일은 만들지 않는다.
    """
    if not candidates:
        raise ValueError("candidates가 비어 있습니다.")
    sampled = gaussian.sample_by_building(train, rows_per_building)
    rows = []
    for params in candidates:
        model = gaussian.train_gaussian(sampled, sampled[common.TARGET], **params)
        _append_score(rows, params, valid[common.TARGET], model.predict(valid))
    return summarize(rows)


def model_file(model_name, building_type, building_id=None):
    """저장 파일 경로를 만든다.

    5_models/{모델}/{건물유형}.joblib
    """
    if building_type not in common.BUILDING_TYPES:
        raise ValueError(f"건물유형은 {common.BUILDING_TYPES} 중 하나여야 합니다.")
    folder = SAVED_MODELS_DIR / model_name
    if building_id is None:
        return folder / f"{building_type}.joblib"
    return folder / building_type / f"{int(building_id)}.joblib"


def _save(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"저장: {path}")
    return path


def save_tree(train, params, building_type):
    """가장 좋은 결정트리 파라미터로 학습 데이터만 다시 학습해 저장한다."""
    model = tree.train_tree(train[TREE_COLUMNS], train[common.TARGET], **params)
    return _save(model, model_file(tree.MODEL_NAME, building_type))


def save_bayesian(train, params, building_type):
    """가장 좋은 베이지안 파라미터로 학습 데이터만 다시 학습해 저장한다."""
    model = bayesian.train_bayesian(train, train[common.TARGET], **params)
    return _save(model, model_file(BAYESIAN_NAME, building_type))


def save_gaussian(train, params, building_type, rows_per_building=gaussian.ROWS_PER_BUILDING):
    """가장 좋은 가우시안 파라미터로 유형 하나의 모델을 학습해 저장한다."""
    sampled = gaussian.sample_by_building(train, rows_per_building)
    model = gaussian.train_gaussian(sampled, sampled[common.TARGET], **params)
    return _save(model, model_file(GAUSSIAN_NAME, building_type))
