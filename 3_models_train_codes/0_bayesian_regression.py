# 베이지안 선형 회귀 학습
# 사후분포 계산은 reference/Bayesian/3_bayesian_regression.py 를 그대로 가져왔다.
# 학습만 하고, 데이터 분할·점수 계산·파일 저장은 하지 않는다.
#
# 호출 예시
#   import importlib
#   bayesian = importlib.import_module("0_bayesian_regression")
#   model = bayesian.train_bayesian(X, y)
#   model = bayesian.train_bayesian(X, y, alpha=1e-6, beta=1e-4)
#   mean, std = model.predict(X_new, return_std=True)

import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge

# 팀 전처리 문서의 입력 명세 (일조·일사는 test.csv에 없어서 제외)
CONTINUOUS_COLUMNS = [
    "temperature",
    "month",
    "total_area",
    "cooling_area",
    "solar_capacity",
    "ess_capacity",
    "pcs_capacity",
]
# 시간대 4구간: 새벽 0-5, 오전 6-11, 오후 12-17, 저녁 18-23
TIME_BINS = [0, 6, 12, 18, 24]
TIME_LABELS = ["dawn", "morning", "afternoon", "evening"]
# 새벽을 기준 범주로 두고 나머지 3개만 원-핫 열로 만든다.
# 4개를 모두 넣으면 네 열의 합이 항상 1이라 절편과 완전히 겹친다(다중공선성).
TIME_DUMMY_COLUMNS = ["time_morning", "time_afternoon", "time_evening"]

Z_95 = 1.96  # 정규분포 95% 구간의 z값


# ---------------------------------------------------------------
# 1. 입력 만들기
# ---------------------------------------------------------------

def time_dummy_columns(time_labels=None):
    labels = TIME_LABELS if time_labels is None else list(time_labels)
    return [f"time_{label}" for label in labels[1:]]


# 시간대 원-핫 인코딩 함수
# 0~3 숫자를 그대로 넣으면 "시간대 번호가 클수록 사용량이 커진다"는 직선 하나로 학습된다.
# 원-핫으로 바꾸면 시간대마다 따로 높낮이(계수)를 가질 수 있다.
def add_time_dummies(data, time_bins=None, time_labels=None):
    bins = TIME_BINS if time_bins is None else time_bins
    labels = TIME_LABELS if time_labels is None else list(time_labels)
    converted = data.copy()
    time_band = pd.cut(converted["hour"], bins=bins, labels=labels, right=False)
    for label in labels[1:]:
        converted[f"time_{label}"] = (time_band == label).astype(np.float64)
    return converted


# 주말 여부 함수 (weekday 5=토, 6=일)
def add_weekend(data):
    converted = data.copy()
    converted["is_weekend"] = converted["weekday"].isin([5, 6]).astype(np.float64)
    return converted


# 입력 행렬 생성 함수
def make_features(data, continuous_columns=None, time_bins=None, time_labels=None):
    columns = CONTINUOUS_COLUMNS if continuous_columns is None else list(continuous_columns)
    labels = TIME_LABELS if time_labels is None else list(time_labels)
    converted = add_weekend(add_time_dummies(data, time_bins, labels))
    binary_columns = ["is_weekend"] + time_dummy_columns(labels)
    return converted[columns + binary_columns].astype(np.float64)


# 값이 모두 같은 열 제외 함수
# 아파트는 설비 3개 열, IDC·상용·호텔은 ESS·PCS가 전부 0이다.
# 표준편차가 0이면 표준화할 때 0으로 나누게 되고, 정보도 없으므로 학습 데이터 기준으로 뺀다.
def drop_constant_columns(train_x, valid_x):
    constant = [column for column in train_x.columns if train_x[column].nunique() <= 1]
    return train_x.drop(columns=constant), valid_x.drop(columns=constant), constant


# 표준화 함수
# 평균·표준편차는 학습 데이터에서만 구하고 검증 데이터에는 같은 값을 적용만 한다.
# 검증 데이터의 통계를 쓰면 미래 정보를 미리 보는 셈(데이터 누수)이 된다.
def standardize(train_x, valid_x, continuous_columns=None):
    source = CONTINUOUS_COLUMNS if continuous_columns is None else list(continuous_columns)
    columns = [column for column in source if column in train_x.columns]
    train_scaled, valid_scaled = train_x.copy(), valid_x.copy()
    if not columns:
        return train_scaled, valid_scaled
    mean = train_x[columns].mean()
    std = train_x[columns].std(ddof=0)
    train_scaled[columns] = (train_x[columns] - mean) / std
    valid_scaled[columns] = (valid_x[columns] - mean) / std
    return train_scaled, valid_scaled


def _is_raw_frame(data):
    """hour, weekday가 있으면 원본 건물 데이터로 보고 입력 변환을 적용한다."""
    return isinstance(data, pd.DataFrame) and {"hour", "weekday"}.issubset(data.columns)


# ---------------------------------------------------------------
# 2. 베이지안 선형 회귀 (강의 식 그대로 numpy로 구현)
# ---------------------------------------------------------------

# α, β 추정 함수
# 주의: sklearn의 이름은 강의와 반대다.
#   sklearn lambda_ = 강의의 α (가중치 정밀도, 사전분포가 얼마나 0 근처로 당기는지)
#   sklearn alpha_  = 강의의 β (노이즈 정밀도, 1/β가 관측 노이즈 분산)
def estimate_alpha_beta(train_x, train_y, fit_intercept=True, max_iter=None, tol=None):
    kwargs = {"fit_intercept": fit_intercept}
    if max_iter is not None:
        kwargs["max_iter"] = max_iter
    if tol is not None:
        kwargs["tol"] = tol
    model = BayesianRidge(**kwargs)
    model.fit(train_x, train_y)
    alpha = model.lambda_
    beta = model.alpha_
    return alpha, beta, model


# 중심화 함수
# 절편(평균 수준)은 학습 데이터의 평균으로 처리하고, 나머지 가중치에만 사전분포 N(0, α⁻¹I)를 준다.
# 절편까지 0으로 당기면 수천 kWh인 평균 수준이 깎이므로, sklearn BayesianRidge도 이렇게 한다.
def center(train_x, train_y):
    x_mean = train_x.mean(axis=0)
    y_mean = train_y.mean()
    return x_mean, y_mean


# 사후 분포 계산 함수
#   S_N⁻¹ = αI + βΦᵀΦ
#   m_N   = β · S_N · Φᵀy
def fit_posterior(phi, t, alpha, beta):
    n_features = phi.shape[1]
    s_n_inv = alpha * np.eye(n_features) + beta * phi.T @ phi
    s_n = np.linalg.inv(s_n_inv)  # 입력이 10개 안팎이라 역행렬을 직접 구해도 빠르다.
    m_n = beta * s_n @ phi.T @ t
    return m_n, s_n


# 예측 분포 계산 함수
#   예측 평균: φᵀ m_N
#   예측 분산: 1/β + φᵀ S_N φ
#     1/β       : 데이터 자체의 노이즈 (아무리 많이 학습해도 줄지 않음)
#     φᵀ S_N φ  : 가중치를 확실히 모르는 데서 오는 불확실성 (데이터가 많을수록 줄어듦)
def predict_distribution(phi, m_n, s_n, beta):
    mean = phi @ m_n
    # φᵀ S_N φ 를 모든 행에 대해 한 번에 계산 (행마다 반복문을 돌지 않기 위함)
    variance = 1.0 / beta + np.sum((phi @ s_n) * phi, axis=1)
    return mean, variance


# 95% 예측구간 포함률 함수 (베이지안 전용 지표)
# 실제값이 [평균 - 1.96σ, 평균 + 1.96σ] 안에 들어간 비율. 0.95에 가까울수록 불확실성을 정직하게 추정한 것.
def coverage_95(y_true, mean, std):
    lower = mean - Z_95 * std
    upper = mean + Z_95 * std
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


class BayesianLinearModel:
    """학습이 끝난 베이지안 선형 회귀.

    predict는 예측 평균(kWh)을 반환한다.
    return_std=True이면 (평균, 표준편차)를 함께 반환한다.
    """

    def _transform(self, X):
        if self.input_mode == "raw":
            features = make_features(
                X, self.continuous_columns, self.time_bins, self.time_labels
            )
            features = features.drop(columns=self.dropped_columns, errors="ignore")
            features = features[self.feature_columns]
            if self.scale_columns:
                features = features.copy()
                features[self.scale_columns] = (
                    features[self.scale_columns] - self.scale_mean
                ) / self.scale_std
            return features.to_numpy(dtype=np.float64)
        if isinstance(X, pd.DataFrame):
            if set(self.feature_columns).issubset(X.columns):
                return X[self.feature_columns].to_numpy(dtype=np.float64)
            return X.to_numpy(dtype=np.float64)
        return np.asarray(X, dtype=np.float64)

    def predict(self, X, return_std=False):
        phi = self._transform(X) - self.x_mean
        mean, variance = predict_distribution(phi, self.m_n, self.s_n, self.beta)
        mean = mean + self.y_mean
        if return_std:
            return mean, np.sqrt(variance)
        return mean


def train_bayesian(
    X,
    y,
    alpha=None,
    beta=None,
    fit_intercept=True,
    continuous_columns=None,
    time_bins=None,
    time_labels=None,
    max_iter=None,
    tol=None,
):
    """전달받은 입력, 정답, 파라미터로 베이지안 선형 회귀를 학습해 반환한다.

    X가 hour, weekday를 가진 데이터프레임이면 원본과 같이
    시간대 원-핫, 주말 여부, 상수 열 제거, 연속형 표준화까지 학습 데이터 기준으로 맞춘다.
    이미 숫자로 만든 행렬이면 그 행렬로 사후분포만 계산한다.

    alpha, beta를 생략하면 sklearn BayesianRidge로 추정한다.
    alpha는 가중치 정밀도(sklearn lambda_), beta는 노이즈 정밀도(sklearn alpha_)다.
    둘 중 하나만 주면 나머지는 추정값을 쓴다.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    continuous_columns = (
        list(CONTINUOUS_COLUMNS) if continuous_columns is None else list(continuous_columns)
    )
    time_bins = TIME_BINS if time_bins is None else time_bins
    time_labels = TIME_LABELS if time_labels is None else list(time_labels)

    model = BayesianLinearModel()
    model.continuous_columns = continuous_columns
    model.time_bins = time_bins
    model.time_labels = time_labels
    model.fit_intercept = fit_intercept

    if _is_raw_frame(X):
        model.input_mode = "raw"
        features = make_features(X, continuous_columns, time_bins, time_labels)
        features, _, dropped = drop_constant_columns(features, features)
        model.dropped_columns = dropped
        model.feature_columns = list(features.columns)
        model.scale_columns = [
            column for column in continuous_columns if column in features.columns
        ]
        if model.scale_columns:
            model.scale_mean = features[model.scale_columns].mean()
            model.scale_std = features[model.scale_columns].std(ddof=0)
        else:
            model.scale_mean = pd.Series(dtype=np.float64)
            model.scale_std = pd.Series(dtype=np.float64)
        features, _ = standardize(features, features, continuous_columns)
        phi = features.to_numpy(dtype=np.float64)
    else:
        model.input_mode = "matrix"
        model.dropped_columns = []
        model.scale_columns = []
        model.scale_mean = None
        model.scale_std = None
        if isinstance(X, pd.DataFrame):
            model.feature_columns = list(X.columns)
            phi = X.to_numpy(dtype=np.float64)
        else:
            phi = np.asarray(X, dtype=np.float64)
            if phi.ndim == 1:
                phi = phi.reshape(-1, 1)
            model.feature_columns = [f"x{i}" for i in range(phi.shape[1])]

    if len(phi) != len(y):
        raise ValueError(f"X와 y의 행 수가 다릅니다: {len(phi)} / {len(y)}")

    if alpha is None or beta is None:
        est_alpha, est_beta, sk_model = estimate_alpha_beta(
            phi, y, fit_intercept=fit_intercept, max_iter=max_iter, tol=tol
        )
        if alpha is None:
            alpha = est_alpha
        if beta is None:
            beta = est_beta
        model.sk_model = sk_model
    else:
        model.sk_model = None

    if fit_intercept:
        x_mean, y_mean = center(phi, y)
    else:
        x_mean = np.zeros(phi.shape[1], dtype=np.float64)
        y_mean = 0.0

    m_n, s_n = fit_posterior(phi - x_mean, y - y_mean, alpha, beta)
    model.alpha = float(alpha)
    model.beta = float(beta)
    model.x_mean = np.asarray(x_mean, dtype=np.float64)
    model.y_mean = float(y_mean)
    model.m_n = m_n
    model.s_n = s_n
    model.weights = pd.Series(m_n, index=model.feature_columns)
    return model
