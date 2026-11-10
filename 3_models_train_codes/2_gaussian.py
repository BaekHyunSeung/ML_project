# 가우시안 프로세스 회귀 학습
# 커널·로그 변환·추세 처리는 reference/가우시안_송이/source_code_file/3_gaussian_model.py 를 그대로 가져왔다.
# 학습만 하고, 데이터 분할·점수 계산·파일 저장·제출 파일 생성은 하지 않는다.
# 원본처럼 건물 1동 분량을 한 번 호출에 넣는 기준으로 둔다. 유형 전체를 한 번에 넣으면 n³으로 느려진다.
#
# 호출 예시
#   import importlib
#   gaussian = importlib.import_module("2_gaussian")
#   model = gaussian.train_gaussian(X, y)
#   model = gaussian.train_gaussian(X, y, trend="none", restarts=0)
#   predicted = model.predict(X_new)

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

# 강수량처럼 영향이 거의 없는 입력은 길이척도가 상한(1e3)에 닿는데, 그때 나오는 경고는 결과에 문제가 없으므로 숨긴다.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

TARGET = "power_consumption"
TRAIN_START = pd.Timestamp("2024-06-01")
HOLIDAYS = pd.to_datetime(["2024-06-06", "2024-08-15"])  # 현충일, 광복절
TREND_MIN_DAYS = 30.0  # trend="smooth" 에서 day_index 길이척도 하한(일)
TRENDS = ("none", "smooth", "linear", "v1")

# 건물 안에서 시간마다 변하는 값만 입력으로 쓴다. 일조·일사는 test에 없어서 제외한다.
WEATHER_COLUMNS = ["temperature", "precipitation", "wind_speed", "humidity"]
TIME_COLUMNS = ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]
RANDOM_STATE = 42


def feature_columns(trend):
    """추세 처리 방식에 따라 입력 열 목록을 돌려준다."""
    if trend == "v1":
        return WEATHER_COLUMNS + TIME_COLUMNS + ["is_weekend", "day_index"]
    base = WEATHER_COLUMNS + TIME_COLUMNS + ["is_off_day"]
    if trend == "smooth":
        return base + ["day_index"]
    return base  # none, linear


def _is_raw_frame(data):
    """원본 CSV처럼 시각 열이 있으면 시간 특성을 만들어서 학습한다."""
    if not isinstance(data, pd.DataFrame):
        return False
    return {"hour", "weekday", "num_date_time"}.issubset(data.columns)


# ---------------------------------------------------------------- 데이터 준비
def parse_date_time(data):
    """num_date_time('11_20240601 00')에서 날짜·시간을 복원한다."""
    raw = data["num_date_time"].str.split("_", n=1).str[1]
    return pd.to_datetime(raw, format="%Y%m%d %H")


def add_time_features(data, holidays=None, train_start=None):
    """시간 열을 주기 함수로 바꾸고 쉬는 날·추세 변수를 만든다."""
    holiday_index = HOLIDAYS if holidays is None else pd.to_datetime(holidays)
    origin = TRAIN_START if train_start is None else pd.Timestamp(train_start)
    converted = data.copy()
    date_time = parse_date_time(converted)
    hour = converted["hour"].to_numpy()
    weekday = converted["weekday"].to_numpy()
    is_weekend = weekday >= 5
    is_holiday = date_time.dt.normalize().isin(holiday_index).to_numpy()
    converted["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    converted["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    converted["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    converted["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    converted["is_weekend"] = is_weekend.astype(np.float64)
    converted["is_holiday"] = is_holiday.astype(np.float64)
    converted["is_off_day"] = (is_weekend | is_holiday).astype(np.float64)
    converted["day_index"] = (date_time - origin).dt.total_seconds().to_numpy() / 86400.0
    return converted


# ---------------------------------------------------------------- 추세(linear 방식)
def fit_linear_trend(day_index, y_log):
    """log(전력) = a + b·day_index 직선을 맞춘다."""
    slope, intercept = np.polyfit(day_index, y_log, 1)
    return slope, intercept


def linear_trend(day_index, coef):
    slope, intercept = coef
    return slope * day_index + intercept


# ---------------------------------------------------------------- 모델
def make_kernel(
    columns,
    scaler,
    trend,
    length_scale_bounds=(1e-2, 1e3),
    constant_value=1.0,
    constant_value_bounds=(1e-3, 1e3),
    noise_level=0.1,
    noise_level_bounds=(1e-6, 1e1),
    trend_min_days=TREND_MIN_DAYS,
):
    """상수 × RBF(입력마다 길이척도 따로) + 백색잡음.
    smooth 방식이면 day_index 길이척도 하한을 30일(표준화 단위로 환산)로 둔다."""
    n = len(columns)
    if (
        isinstance(length_scale_bounds, tuple)
        and len(length_scale_bounds) == 2
        and isinstance(length_scale_bounds[0], (int, float))
    ):
        bounds = [length_scale_bounds] * n
    else:
        bounds = [tuple(bound) for bound in length_scale_bounds]
    init = np.ones(n)
    if trend == "smooth":
        i = columns.index("day_index")
        lower = trend_min_days / float(scaler.scale_[i])
        bounds[i] = (lower, bounds[i][1])
        init[i] = max(1.0, lower)
    return (
        ConstantKernel(constant_value, constant_value_bounds)
        * RBF(length_scale=init, length_scale_bounds=bounds)
        + WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bounds)
    )


def fit_gpr(
    x_train,
    y_log,
    kernel,
    restarts,
    optimize=True,
    normalize_y=True,
    random_state=RANDOM_STATE,
):
    """입력은 표준화한 상태로 받는다. 출력(y_log)은 log1p 변환(및 추세 제거)이 끝난 값."""
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=normalize_y,
        n_restarts_optimizer=restarts if optimize else 0,
        optimizer="fmin_l_bfgs_b" if optimize else None,
        random_state=random_state,
    )
    model.fit(x_train, y_log)
    return model


class GaussianProcessModel:
    """학습이 끝난 가우시안 프로세스.

    predict는 kWh 단위 예측을 반환한다. 학습 때 log1p로 맞춘 값을 expm1로 되돌리고 0 미만은 자른다.
    """

    def _design(self, X):
        if self.input_mode == "raw":
            if not isinstance(X, pd.DataFrame):
                raise TypeError("이 모델은 num_date_time, hour, weekday가 있는 데이터프레임으로 예측합니다.")
            frame = add_time_features(X, self.holidays, self.train_start)
            missing = [column for column in self.columns if column not in frame.columns]
            if self.trend == "linear" and "day_index" not in frame.columns:
                missing.append("day_index")
            if missing:
                raise KeyError(f"가우시안 입력 열이 없습니다: {missing}")
            return frame, frame[self.columns].to_numpy(dtype=np.float64)
        if isinstance(X, pd.DataFrame):
            if all(column in X.columns for column in self.columns):
                return X, X[self.columns].to_numpy(dtype=np.float64)
            return None, X.to_numpy(dtype=np.float64)
        return None, np.asarray(X, dtype=np.float64)

    def predict(self, X):
        frame, features = self._design(X)
        offset = 0.0
        if self.trend == "linear":
            offset = linear_trend(frame["day_index"].to_numpy(dtype=np.float64), self.trend_coef)
        log_pred = self.model.predict(self.scaler.transform(features)) + offset
        return np.clip(np.expm1(log_pred), 0.0, None)


def train_gaussian(
    X,
    y,
    trend="none",
    restarts=1,
    random_state=RANDOM_STATE,
    normalize_y=True,
    kernel=None,
    length_scale_bounds=(1e-2, 1e3),
    constant_value=1.0,
    constant_value_bounds=(1e-3, 1e3),
    noise_level=0.1,
    noise_level_bounds=(1e-6, 1e1),
    trend_min_days=TREND_MIN_DAYS,
    holidays=None,
    train_start=None,
):
    """전달받은 입력, 정답, 파라미터로 가우시안 프로세스 회귀를 학습해 반환한다.

    X가 num_date_time, hour, weekday를 가진 데이터프레임이면 원본과 같이
    주기 함수·쉬는 날·추세 변수를 만든 뒤, 날씨와 시간 열만 표준화해서 학습한다.
    이미 숫자로 만든 행렬이면 trend는 "none"만 쓸 수 있고, 그 행렬을 표준화해서 학습한다.

    y는 kWh이다. 안에서 log1p를 씌워 학습한다.
    kernel을 주면 하이퍼파라미터 최적화 없이 그 커널을 고정한 채 사후분포만 맞춘다.
    trend: none(기본, 추세 열 없음), smooth(day_index 길이척도 하한), linear(직선 추세 제거), v1(비교용).
    """
    if trend not in TRENDS:
        raise ValueError(f"trend는 {TRENDS} 중 하나여야 합니다.")

    y = np.asarray(y, dtype=np.float64).reshape(-1)
    holidays = HOLIDAYS if holidays is None else pd.to_datetime(holidays)
    train_start = TRAIN_START if train_start is None else pd.Timestamp(train_start)

    if _is_raw_frame(X):
        frame = add_time_features(X, holidays, train_start)
        columns = feature_columns(trend)
        missing = [column for column in columns if column not in frame.columns]
        if trend == "linear" and "day_index" not in frame.columns:
            missing.append("day_index")
        if missing:
            raise KeyError(f"가우시안 입력 열이 없습니다: {missing}")
        input_mode = "raw"
    else:
        if trend != "none":
            raise ValueError(
                "숫자 행렬은 trend='none'으로 학습합니다. "
                "smooth, linear, v1은 num_date_time, hour, weekday가 있는 데이터프레임이 필요합니다."
            )
        if isinstance(X, pd.DataFrame):
            columns = list(X.columns)
            frame = X
        else:
            matrix = np.asarray(X, dtype=np.float64)
            if matrix.ndim == 1:
                matrix = matrix.reshape(-1, 1)
            columns = [f"x{i}" for i in range(matrix.shape[1])]
            frame = pd.DataFrame(matrix, columns=columns)
        input_mode = "matrix"

    if len(frame) != len(y):
        raise ValueError(f"X와 y의 행 수가 다릅니다: {len(frame)} / {len(y)}")

    x_fit = frame[columns].to_numpy(dtype=np.float64)
    y_log = np.log1p(y)
    offset = np.zeros(len(y), dtype=np.float64)
    trend_coef = None
    if trend == "linear":
        day_index = frame["day_index"].to_numpy(dtype=np.float64)
        trend_coef = fit_linear_trend(day_index, y_log)
        offset = linear_trend(day_index, trend_coef)

    scaler = StandardScaler().fit(x_fit)
    optimize = kernel is None
    if kernel is None:
        kernel = make_kernel(
            columns,
            scaler,
            trend,
            length_scale_bounds=length_scale_bounds,
            constant_value=constant_value,
            constant_value_bounds=constant_value_bounds,
            noise_level=noise_level,
            noise_level_bounds=noise_level_bounds,
            trend_min_days=trend_min_days,
        )

    gpr = fit_gpr(
        scaler.transform(x_fit),
        y_log - offset,
        kernel,
        restarts,
        optimize=optimize,
        normalize_y=normalize_y,
        random_state=random_state,
    )

    model = GaussianProcessModel()
    model.model = gpr
    model.kernel_ = gpr.kernel_
    model.scaler = scaler
    model.columns = list(columns)
    model.trend = trend
    model.trend_coef = trend_coef
    model.input_mode = input_mode
    model.holidays = holidays
    model.train_start = train_start
    model.restarts = restarts
    model.random_state = random_state
    return model
