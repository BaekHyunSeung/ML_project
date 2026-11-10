# 가우시안 프로세스 회귀 모델 (v2)
# source_data의 건물유형별 훈련 CSV를 읽어 건물마다 가우시안 프로세스 회귀(GPR) 모델을 학습한다.
# 1) 훈련 기간을 날짜 기준 80/20으로 나눠(6/1~8/7 학습, 8/8~8/24 검증) 성능(SMAPE, MAE, RMSE)을 기록한다.
# 2) 검증에서 찾은 커널 하이퍼파라미터로 전체 기간을 다시 학습해 test 기간(8/25~8/31)을 예측한다.
#
# v1 → v2 변경점
#   - 공휴일 변수: 6/6 현충일, 8/15 광복절(둘 다 목요일)을 주말과 합쳐 is_off_day로 쓴다.
#     v1은 공휴일이 없어서 "목요일은 가끔 낮다"로 잘못 배웠다.
#   - 추세 변수(day_index) 처리를 --trend 옵션으로 고른다. v1은 RBF 길이척도가 8~10일로 짧게 학습되어
#     훈련 마지막 날 이후 예측이 평균 쪽으로 미끄러졌다.
#       none   : day_index를 뺀다. 날씨·시간만으로 학습
#       smooth : day_index를 쓰되 길이척도 하한을 30일로 둬서 완만한 추세만 담는다
#       linear : log(전력)을 day_index에 대해 직선으로 먼저 맞추고(추세), 그 잔차를 GP로 학습한다
#       v1     : v1과 같은 설정(비교용)
#
# 실행 예)
#   python 3_gaussian_model.py apartment                         # 건물유형 하나 (기본 --trend none)
#   python 3_gaussian_model.py apartment department_store --trend none --restarts 0 --tag cmp_none
#   python 3_gaussian_model.py all                               # 10개 전부 + 제출 파일 합치기
#
# 가우시안 프로세스는 데이터 수의 세제곱에 비례해 느려지므로, 유형 전체(2만 행)가 아니라
# 건물 1동(2,040행)마다 모델을 하나씩 만든다. 면적·설비 용량은 건물 안에서 상수라 입력에서 뺀다.

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler

# 강수량처럼 영향이 거의 없는 입력은 길이척도가 상한(1e3)에 닿는데, 그때 나오는 경고는 결과에 문제가 없으므로 숨긴다.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_DIR / "source_data" / "train"
TEST_DIR = PROJECT_DIR / "source_data" / "test"
OUTPUT_ROOT = PROJECT_DIR / "model_output"

BUILDING_TYPES = [
    "apartment",
    "commercial",
    "department_store",
    "hospital",
    "hotel",
    "idc",
    "other",
    "public",
    "research",
    "school",
]

TARGET = "power_consumption"
TRAIN_START = pd.Timestamp("2024-06-01")
VALID_START = pd.Timestamp("2024-08-08")  # 이 시각 미만은 학습(6/1~8/7, 68일, 80%), 이후는 검증(8/8~8/24, 17일, 20%). 팀 공통 기준
# 이전 v2까지는 마지막 7일(8/18~)만 검증했다. --valid-start 2024-08-18 로 재현 가능
HOLIDAYS = pd.to_datetime(["2024-06-06", "2024-08-15"])  # 현충일, 광복절. test 기간(8/25~31)에는 공휴일 없음
TREND_MIN_DAYS = 30.0  # --trend smooth 에서 day_index 길이척도 하한(일)

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


# ---------------------------------------------------------------- 데이터 준비
def load_type(building_type):
    train = pd.read_csv(TRAIN_DIR / f"{building_type}.csv")
    test = pd.read_csv(TEST_DIR / f"{building_type}.csv")
    return train, test


def parse_date_time(data):
    """num_date_time('11_20240601 00')에서 날짜·시간을 복원한다."""
    raw = data["num_date_time"].str.split("_", n=1).str[1]
    return pd.to_datetime(raw, format="%Y%m%d %H")


def add_time_features(data):
    """시간 열을 주기 함수로 바꾸고 쉬는 날·추세 변수를 만든다."""
    converted = data.copy()
    date_time = parse_date_time(converted)
    hour = converted["hour"].to_numpy()
    weekday = converted["weekday"].to_numpy()
    is_weekend = weekday >= 5
    is_holiday = date_time.dt.normalize().isin(HOLIDAYS).to_numpy()
    converted["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    converted["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    converted["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    converted["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    converted["is_weekend"] = is_weekend.astype(np.float64)
    converted["is_holiday"] = is_holiday.astype(np.float64)
    converted["is_off_day"] = (is_weekend | is_holiday).astype(np.float64)
    converted["day_index"] = (date_time - TRAIN_START).dt.total_seconds().to_numpy() / 86400.0
    return converted


# ---------------------------------------------------------------- 평가 지표
def smape(actual, predicted):
    """대회 평가 지표. 0~200 사이 값, 작을수록 좋다."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    safe = np.where(denominator == 0, 1, denominator)
    ratio = np.where(denominator == 0, 0.0, np.abs(actual - predicted) / safe)
    return float(np.mean(ratio) * 100)


def mae(actual, predicted):
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


# ---------------------------------------------------------------- 추세(linear 방식)
def fit_linear_trend(day_index, y_log):
    """log(전력) = a + b·day_index 직선을 맞춘다."""
    slope, intercept = np.polyfit(day_index, y_log, 1)
    return slope, intercept


def linear_trend(day_index, coef):
    slope, intercept = coef
    return slope * day_index + intercept


# ---------------------------------------------------------------- 모델
def make_kernel(columns, scaler, trend):
    """상수 × RBF(입력마다 길이척도 따로) + 백색잡음.
    smooth 방식이면 day_index 길이척도 하한을 30일(표준화 단위로 환산)로 둔다."""
    n = len(columns)
    bounds = [(1e-2, 1e3)] * n
    init = np.ones(n)
    if trend == "smooth":
        i = columns.index("day_index")
        lower = TREND_MIN_DAYS / float(scaler.scale_[i])
        bounds[i] = (lower, 1e3)
        init[i] = max(1.0, lower)
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=init, length_scale_bounds=bounds)
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1))
    )


def fit_gpr(x_train, y_log, kernel, restarts, optimize=True):
    """입력은 표준화한 상태로 받는다. 출력(y_log)은 log1p 변환(및 추세 제거)이 끝난 값."""
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=restarts if optimize else 0,
        optimizer="fmin_l_bfgs_b" if optimize else None,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train, y_log)
    return model


def fit_and_predict(fit_part, pred_part, columns, trend, restarts, kernel=None):
    """fit_part로 학습해 pred_part를 예측한다. kernel을 주면 최적화 없이 그 값을 고정해 쓴다.
    돌려주는 값: 예측(kWh), 학습된 커널"""
    x_fit = fit_part[columns].to_numpy(dtype=np.float64)
    x_pred = pred_part[columns].to_numpy(dtype=np.float64)
    y_log = np.log1p(fit_part[TARGET].to_numpy(dtype=np.float64))

    offset_fit = np.zeros(len(fit_part))
    offset_pred = np.zeros(len(pred_part))
    if trend == "linear":
        coef = fit_linear_trend(fit_part["day_index"].to_numpy(), y_log)
        offset_fit = linear_trend(fit_part["day_index"].to_numpy(), coef)
        offset_pred = linear_trend(pred_part["day_index"].to_numpy(), coef)

    scaler = StandardScaler().fit(x_fit)
    if kernel is None:
        kernel = make_kernel(columns, scaler, trend)
        model = fit_gpr(scaler.transform(x_fit), y_log - offset_fit, kernel, restarts, optimize=True)
    else:
        model = fit_gpr(scaler.transform(x_fit), y_log - offset_fit, kernel, restarts, optimize=False)

    log_pred = model.predict(scaler.transform(x_pred)) + offset_pred
    return np.clip(np.expm1(log_pred), 0.0, None), model.kernel_


def run_building(building_id, train, test, trend, restarts, validate_only, valid_start_day):
    """건물 1동: 검증 → 전체 재학습 → test 예측."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)  # 병렬 작업자 프로세스에도 적용
    started = time.time()
    columns = feature_columns(trend)
    train = train.sort_values("num_date_time").reset_index(drop=True)
    test = test.sort_values("num_date_time").reset_index(drop=True)

    valid_start = valid_start_day  # 검증 시작일의 0시 (day_index 단위, 6/1 = 0)
    fit_part = train[train["day_index"] < valid_start]
    valid_part = train[train["day_index"] >= valid_start]
    y_valid = valid_part[TARGET].to_numpy(dtype=np.float64)

    # 1) 검증: 검증 시작일 이전으로 학습, 이후를 예측 (기본 68일 → 17일)
    valid_pred, learned_kernel = fit_and_predict(fit_part, valid_part, columns, trend, restarts)

    # 2) 전체 재학습: 검증에서 찾은 커널 값을 고정하고 85일 전부로 다시 학습 → test 예측
    if validate_only:
        test_pred = np.full(len(test), np.nan)
    else:
        test_pred, _ = fit_and_predict(train, test, columns, trend, restarts, kernel=learned_kernel)

    metrics = {
        "building_id": int(building_id),
        "train_rows": len(fit_part),
        "valid_rows": len(valid_part),
        "smape": round(smape(y_valid, valid_pred), 3),
        "mae": round(mae(y_valid, valid_pred), 2),
        "rmse": round(rmse(y_valid, valid_pred), 2),
        "seconds": round(time.time() - started, 1),
        "kernel": str(learned_kernel),
    }
    valid_frame = pd.DataFrame(
        {
            "num_date_time": valid_part["num_date_time"].to_numpy(),
            "building_id": int(building_id),
            "actual": y_valid,
            "predicted": np.round(valid_pred, 2),
        }
    )
    test_frame = pd.DataFrame(
        {"num_date_time": test["num_date_time"].to_numpy(), "answer": np.round(test_pred, 2)}
    )
    return metrics, valid_frame, test_frame


def run_type(building_type, output_dir, trend, restarts, validate_only, n_jobs, valid_start=VALID_START):
    train, test = load_type(building_type)
    train = add_time_features(train)
    test = add_time_features(test)
    building_ids = sorted(train["building_id"].unique())
    valid_start_day = (pd.Timestamp(valid_start) - TRAIN_START).total_seconds() / 86400.0
    print(f"\n[{building_type}] 건물 {len(building_ids)}동, 훈련 {len(train)}행, test {len(test)}행, 검증 시작 {pd.Timestamp(valid_start).date()}", flush=True)

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_building)(
            building_id,
            train[train["building_id"] == building_id],
            test[test["building_id"] == building_id],
            trend,
            restarts,
            validate_only,
            valid_start_day,
        )
        for building_id in building_ids
    )

    metrics = pd.DataFrame([r[0] for r in results])
    valid_frame = pd.concat([r[1] for r in results], ignore_index=True)
    test_frame = pd.concat([r[2] for r in results], ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / f"{building_type}_validation.csv", index=False)
    valid_frame.to_csv(output_dir / f"{building_type}_validation_pred.csv", index=False)
    if not validate_only:
        test_frame.to_csv(output_dir / f"{building_type}_prediction.csv", index=False)

    print(metrics.drop(columns=["kernel"]).to_string(index=False), flush=True)
    type_smape = smape(valid_frame["actual"], valid_frame["predicted"])
    print(f"{building_type} 전체 SMAPE: {type_smape:.3f}  (건물 평균 {metrics['smape'].mean():.3f})", flush=True)
    return building_type, type_smape, len(building_ids)


def merge_submission(types, output_dir):
    frames = [pd.read_csv(output_dir / f"{t}_prediction.csv") for t in types]
    submission = pd.concat(frames, ignore_index=True)
    # 제출 양식 순서: 건물번호, 일시 순
    key = submission["num_date_time"].str.split("_", n=1, expand=True)
    submission["_building"] = key[0].astype(int)
    submission["_time"] = key[1]
    submission = submission.sort_values(["_building", "_time"]).drop(columns=["_building", "_time"])
    submission.to_csv(output_dir / "gaussian_submission.csv", index=False)
    print(f"\n제출 파일 저장: {output_dir / 'gaussian_submission.csv'} ({len(submission)}행)", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="건물별 가우시안 프로세스 회귀")
    parser.add_argument("types", nargs="*", default=["apartment"], help="건물유형 이름들 또는 all")
    parser.add_argument("--trend", choices=["none", "smooth", "linear", "v1"], default="none", help="추세 변수 처리 방식 (비교 결과 none이 가장 좋아 기본값)")
    parser.add_argument("--restarts", type=int, default=1, help="커널 최적화 재시작 횟수 (0이면 빠름)")
    parser.add_argument("--tag", default="gaussian", help="결과 폴더 이름 (model_output/<tag>)")
    parser.add_argument("--validate-only", action="store_true", help="검증만 하고 test 예측은 생략")
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1), help="동시에 학습할 건물 수")
    parser.add_argument("--valid-start", default=str(VALID_START.date()), help="검증 시작일 (이 날 0시 미만은 학습). 기본 2024-08-08 (80/20)")
    return parser.parse_args()


def main():
    args = parse_args()
    targets = BUILDING_TYPES if args.types == ["all"] else args.types
    unknown = [t for t in targets if t not in BUILDING_TYPES]
    if unknown:
        print(f"모르는 건물유형: {unknown}\n가능한 값: {BUILDING_TYPES} 또는 all")
        return

    output_dir = OUTPUT_ROOT / args.tag
    print(f"설정: trend={args.trend}, restarts={args.restarts}, 검증 시작={args.valid_start}, 입력={feature_columns(args.trend)}", flush=True)
    print(f"결과 폴더: {output_dir}", flush=True)

    summary = [
        run_type(t, output_dir, args.trend, args.restarts, args.validate_only, args.jobs, args.valid_start)
        for t in targets
    ]

    print("\n[건물유형별 검증 SMAPE]")
    for building_type, type_smape, count in summary:
        print(f"  {building_type:17s} {type_smape:7.3f}  ({count}동)")

    if len(targets) == len(BUILDING_TYPES) and not args.validate_only:
        merge_submission(targets, output_dir)
    print(f"\n결과 폴더: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
