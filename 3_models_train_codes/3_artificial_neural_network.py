# 인공신경망 회귀 학습
# 전달받은 입력과 정답으로 학습만 한다.
# 데이터 분할, 점수 계산, 파일 저장은 하지 않는다.
# 입력은 팀 명세(기온, 월, 시간대, 주말, 면적·설비)로 만들고, 학습 통계로 표준화한다.
#
# 호출 예시
#   import importlib
#   neural = importlib.import_module("3_artificial_neural_network")
#   model = neural.train_neural_network(X, y)
#   model = neural.train_neural_network(X, y, hidden_layers="64-32", alpha=1e-3)

import importlib
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

common = importlib.import_module("0_common")

MODEL_NAME = "neural_network"
HIDDEN_LAYERS = "64-32"
L2_ALPHA = 1e-3
LEARNING_RATE = 1e-3
MAX_EPOCHS = 500
PATIENCE = 20
EARLY_STOP_FRACTION = 0.1
RANDOM_STATE = 42


def hidden_layer_sizes(hidden_layers):
    """'64-32' 또는 (64, 32)를 sklearn 은닉층 크기로 바꾼다."""
    if isinstance(hidden_layers, (tuple, list)):
        return tuple(int(size) for size in hidden_layers)
    return tuple(int(size) for size in str(hidden_layers).split("-") if size)


class NeuralNetworkModel:
    """학습 때 맞춘 표준화 기준으로 예측하는 신경망."""

    def predict(self, X):
        features = common.make_base_features(X)[self.columns]
        scaled = features.copy()
        columns = [column for column in self.continuous_columns if column in scaled.columns]
        scaled[columns] = (features[columns] - self.x_mean[columns]) / self.x_std[columns]
        scaled_pred = self.mlp.predict(scaled.to_numpy(dtype=np.float64))
        return scaled_pred * self.y_std + self.y_mean


def train_neural_network(
    X,
    y,
    hidden_layers=HIDDEN_LAYERS,
    alpha=L2_ALPHA,
    learning_rate_init=LEARNING_RATE,
    max_iter=MAX_EPOCHS,
    n_iter_no_change=PATIENCE,
    validation_fraction=EARLY_STOP_FRACTION,
    early_stopping=True,
    random_state=RANDOM_STATE,
):
    """전달받은 입력과 정답으로 인공신경망을 학습해 반환한다.

    인자를 생략하면 은닉층 64-32, L2 0.001, 학습률 0.001, 최대 500에폭,
    조기 종료, random_state 42로 학습한다.
    """
    train_x = common.make_base_features(X)
    constant = [column for column in train_x.columns if train_x[column].nunique() <= 1]
    train_x = train_x.drop(columns=constant)
    continuous = [column for column in common.BASE_CONTINUOUS_COLUMNS if column in train_x.columns]
    x_mean = train_x[continuous].mean()
    x_std = train_x[continuous].std(ddof=0).replace(0, 1)
    train_x = train_x.copy()
    train_x[continuous] = (train_x[continuous] - x_mean) / x_std

    target = pd.Series(np.asarray(y, dtype=np.float64))
    y_mean = float(target.mean())
    y_std = float(target.std(ddof=1))
    if y_std == 0:
        y_std = 1.0
    y_scaled = ((target - y_mean) / y_std).to_numpy()

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes(hidden_layers),
        activation="relu",
        solver="adam",
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp.fit(train_x.to_numpy(dtype=np.float64), y_scaled)

    model = NeuralNetworkModel()
    model.mlp = mlp
    model.columns = list(train_x.columns)
    model.continuous_columns = continuous
    model.x_mean = x_mean
    model.x_std = x_std
    model.y_mean = y_mean
    model.y_std = y_std
    return model
