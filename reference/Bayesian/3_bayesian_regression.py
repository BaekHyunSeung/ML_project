# 베이지안 선형 회귀 (담당: 박준희)
# 건물유형마다 따로 학습하고, 날짜 기준 검증 20%로 평가한다.
# 5주차 '실습 2. 베이지안 선형회귀'와 같은 식으로 사후 분포를 numpy로 직접 계산하고,
# 하이퍼파라미터 α, β만 sklearn BayesianRidge로 데이터에서 추정한다.
#
# 실행 방법
#   python 3_bayesian_regression.py public   → 공공 유형 하나만 실행하고 정상 작동을 점검
#   python 3_bayesian_regression.py          → 10개 유형 전체 실행 + 결과 CSV + 그래프 저장

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge

from common import (
    BUILDING_TYPE_NAMES,
    BUILDING_TYPES,
    RESULTS_DIR,
    TARGET,
    load_train_valid,
    regression_metrics,
    save_results,
    setup_korean_font,
)

MODEL_NAME = "bayesian_linear_regression"

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
BINARY_COLUMNS = ["is_weekend"] + TIME_DUMMY_COLUMNS

Z_95 = 1.96  # 정규분포 95% 구간의 z값


# ---------------------------------------------------------------
# 1. 입력 만들기
# ---------------------------------------------------------------

# 시간대 원-핫 인코딩 함수
# 0~3 숫자를 그대로 넣으면 "시간대 번호가 클수록 사용량이 커진다"는 직선 하나로 학습된다.
# 원-핫으로 바꾸면 시간대마다 따로 높낮이(계수)를 가질 수 있다.
def add_time_dummies(data):
    converted = data.copy()
    time_band = pd.cut(converted["hour"], bins=TIME_BINS, labels=TIME_LABELS, right=False)
    for label in TIME_LABELS[1:]:
        converted[f"time_{label}"] = (time_band == label).astype(np.float64)
    return converted


# 주말 여부 함수 (weekday 5=토, 6=일)
def add_weekend(data):
    converted = data.copy()
    converted["is_weekend"] = converted["weekday"].isin([5, 6]).astype(np.float64)
    return converted


# 입력 행렬 생성 함수
def make_features(data):
    converted = add_weekend(add_time_dummies(data))
    return converted[CONTINUOUS_COLUMNS + BINARY_COLUMNS].astype(np.float64)


# 값이 모두 같은 열 제외 함수
# 아파트는 설비 3개 열, IDC·상용·호텔은 ESS·PCS가 전부 0이다.
# 표준편차가 0이면 표준화할 때 0으로 나누게 되고, 정보도 없으므로 학습 데이터 기준으로 뺀다.
def drop_constant_columns(train_x, valid_x):
    constant = [column for column in train_x.columns if train_x[column].nunique() <= 1]
    return train_x.drop(columns=constant), valid_x.drop(columns=constant), constant


# 표준화 함수
# 평균·표준편차는 학습 데이터에서만 구하고 검증 데이터에는 같은 값을 적용만 한다.
# 검증 데이터의 통계를 쓰면 미래 정보를 미리 보는 셈(데이터 누수)이 된다.
def standardize(train_x, valid_x):
    columns = [column for column in CONTINUOUS_COLUMNS if column in train_x.columns]
    mean = train_x[columns].mean()
    std = train_x[columns].std(ddof=0)
    train_scaled, valid_scaled = train_x.copy(), valid_x.copy()
    train_scaled[columns] = (train_x[columns] - mean) / std
    valid_scaled[columns] = (valid_x[columns] - mean) / std
    return train_scaled, valid_scaled


# ---------------------------------------------------------------
# 2. 베이지안 선형 회귀 (강의 식 그대로 numpy로 구현)
# ---------------------------------------------------------------

# α, β 추정 함수
# 주의: sklearn의 이름은 강의와 반대다.
#   sklearn lambda_ = 강의의 α (가중치 정밀도, 사전분포가 얼마나 0 근처로 당기는지)
#   sklearn alpha_  = 강의의 β (노이즈 정밀도, 1/β가 관측 노이즈 분산)
def estimate_alpha_beta(train_x, train_y):
    model = BayesianRidge(fit_intercept=True)
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


# ---------------------------------------------------------------
# 3. 유형 하나 학습·검증
# ---------------------------------------------------------------

def prepare_inputs(train, valid):
    train_x, valid_x = make_features(train), make_features(valid)
    train_x, valid_x, dropped = drop_constant_columns(train_x, valid_x)
    train_x, valid_x = standardize(train_x, valid_x)
    return train_x, valid_x, dropped


def run_one_type(building_type, verbose=True):
    train, valid = load_train_valid(building_type, verbose=verbose)
    train_x, valid_x, dropped = prepare_inputs(train, valid)
    train_y = train[TARGET].to_numpy(dtype=np.float64)
    valid_y = valid[TARGET].to_numpy(dtype=np.float64)
    if verbose and dropped:
        print(f"  값이 모두 같아 제외한 열: {dropped}")

    # (1) sklearn으로 α, β 추정
    alpha, beta, sk_model = estimate_alpha_beta(train_x.to_numpy(), train_y)

    # (2) 강의 식으로 사후 분포 계산 (중심화한 Φ, t 사용)
    x_mean, y_mean = center(train_x.to_numpy(), train_y)
    phi_train = train_x.to_numpy() - x_mean
    m_n, s_n = fit_posterior(phi_train, train_y - y_mean, alpha, beta)

    # (3) 검증 데이터 예측: 중심화를 되돌리기 위해 학습 평균 y_mean을 더한다.
    phi_valid = valid_x.to_numpy() - x_mean
    mean, variance = predict_distribution(phi_valid, m_n, s_n, beta)
    mean = mean + y_mean
    std = np.sqrt(variance)

    metrics = regression_metrics(valid_y, mean)
    result = {
        "model": MODEL_NAME,
        "building_type": building_type,
        "n_train": len(train),
        "n_valid": len(valid),
        **metrics,
        "coverage95": coverage_95(valid_y, mean, std),
        "alpha": alpha,
        "beta": beta,
    }
    predictions = pd.DataFrame(
        {
            "building_type": building_type,
            "building_id": valid["building_id"].to_numpy(),
            "date_time": valid["date_time"].to_numpy(),
            "actual": valid_y,
            "pred_mean": mean,
            "pred_std": std,
            "lower95": mean - Z_95 * std,
            "upper95": mean + Z_95 * std,
        }
    )
    extras = {
        "sk_model": sk_model,
        "valid_x": valid_x,
        "train_y": train_y,
        "weights": pd.Series(m_n, index=train_x.columns),
    }
    return result, predictions, extras


# ---------------------------------------------------------------
# 4. 정상 작동 점검 (유형 1개로 먼저 확인)
# ---------------------------------------------------------------

def check_one_type(result, predictions, extras, verbose=True):
    mean = predictions["pred_mean"].to_numpy()
    std = predictions["pred_std"].to_numpy()
    actual = predictions["actual"].to_numpy()

    no_nan = not (np.isnan(mean).any() or np.isnan(std).any())
    same_count = len(mean) == result["n_valid"]

    # 단순 평균 예측: 학습 데이터 평균 하나로 모든 시간을 예측했을 때
    baseline = np.full_like(actual, extras["train_y"].mean())
    baseline_metrics = regression_metrics(actual, baseline)
    beats_baseline = result["RMSE"] < baseline_metrics["RMSE"]

    # numpy 구현과 sklearn 예측 비교
    sk_mean, sk_std = extras["sk_model"].predict(extras["valid_x"].to_numpy(), return_std=True)
    max_mean_diff = np.max(np.abs(sk_mean - mean))
    max_std_diff = np.max(np.abs(sk_std - std))
    matches_sklearn = np.allclose(sk_mean, mean, rtol=1e-6, atol=1e-3) and np.allclose(
        sk_std, std, rtol=1e-6, atol=1e-3
    )

    ok = all([no_nan, same_count, beats_baseline, matches_sklearn])
    if not verbose:
        return ok

    print("\n[정상 작동 점검]")
    print(f"  NaN 없음: {no_nan}")
    print(f"  예측 개수 = 검증 행 수: {same_count} ({len(mean)} / {result['n_valid']})")
    print(
        f"  지표: MAE {result['MAE']:.2f}, RMSE {result['RMSE']:.2f}, "
        f"R² {result['R2']:.4f}, 95% 포함률 {result['coverage95']:.3f}"
    )
    print(
        f"  단순 평균 예측: MAE {baseline_metrics['MAE']:.2f}, RMSE {baseline_metrics['RMSE']:.2f}, "
        f"R² {baseline_metrics['R2']:.4f} → 베이지안이 더 나음: {beats_baseline}"
    )
    print(
        f"  numpy vs sklearn: 평균 최대 차이 {max_mean_diff:.2e}, 표준편차 최대 차이 {max_std_diff:.2e} "
        f"→ 일치: {matches_sklearn}"
    )
    print(f"  추정값: α(가중치 정밀도) {result['alpha']:.4e}, β(노이즈 정밀도) {result['beta']:.4e}")
    print(f"  노이즈 표준편차 1/√β = {1 / np.sqrt(result['beta']):.1f} kWh")
    # 참고: 면적·설비 열은 건물마다 하나의 값이라 유형 안 건물 수(8~16동)만큼만 다르다.
    # ESS와 PCS처럼 건물 간 상관이 거의 1인 열은 가중치가 서로 크게 상쇄될 수 있어서
    # 개별 가중치 크기보다 예측 결과로 해석해야 한다.
    print("  사후 평균 가중치 m_N (표준화된 입력 기준):")
    print(extras["weights"].round(2).to_string())
    return ok


# ---------------------------------------------------------------
# 5. 시각화
# ---------------------------------------------------------------

# 대표 건물 선택 함수: 검증 기간 평균 사용량이 유형 안에서 중앙값에 가장 가까운 건물
# (가장 크거나 작은 건물이 아니라 "그 유형의 전형적인 건물"을 보여주기 위함)
def pick_representative_building(predictions):
    building_mean = predictions.groupby("building_id")["actual"].mean()
    median = building_mean.median()
    return int((building_mean - median).abs().idxmin())


def plot_prediction_band(predictions, building_type):
    building_id = pick_representative_building(predictions)
    part = predictions[predictions["building_id"] == building_id].sort_values("date_time")

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.fill_between(part["date_time"], part["lower95"], part["upper95"],
                    color="tab:blue", alpha=0.2, label="95% 예측구간")
    ax.plot(part["date_time"], part["actual"], color="black", linewidth=1, label="실제값")
    ax.plot(part["date_time"], part["pred_mean"], color="tab:blue", linewidth=1.2, label="예측 평균")
    name = BUILDING_TYPE_NAMES[building_type]
    ax.set_title(f"{name} - {building_id}번 건물 검증 기간 예측 (베이지안 선형 회귀)")
    ax.set_xlabel("일시")
    ax.set_ylabel("전력소비량(kWh)")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = RESULTS_DIR / f"bayesian_band_{building_type}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return building_id


def plot_r2_bar(results):
    table = results.sort_values("R2", ascending=False)
    labels = [BUILDING_TYPE_NAMES[t] for t in table["building_type"]]
    colors = ["tab:blue" if r2 >= 0 else "tab:red" for r2 in table["R2"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, table["R2"], color=colors)
    ax.bar_label(bars, fmt="%.2f", padding=2)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title("건물유형별 검증 R² (베이지안 선형 회귀)")
    ax.set_ylabel("R²")
    ax.set_xlabel("건물유형")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "bayesian_r2_by_type.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------
# 6. 실행
# ---------------------------------------------------------------

def run_single(building_type):
    result, predictions, extras = run_one_type(building_type)
    ok = check_one_type(result, predictions, extras)
    print(f"\n점검 결과: {'모두 통과' if ok else '확인 필요'}")


def run_all():
    setup_korean_font()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows, all_predictions = [], []
    for building_type in BUILDING_TYPES:
        result, predictions, extras = run_one_type(building_type)
        # 10개 유형 모두 같은 점검(NaN, 개수, 평균 예측 대비, sklearn 일치)을 조용히 수행한다.
        ok = check_one_type(result, predictions, extras, verbose=False)
        rows.append(result)
        all_predictions.append(predictions)
        building_id = plot_prediction_band(predictions, building_type)
        print(
            f"  → R² {result['R2']:.3f}, MAE {result['MAE']:.1f}, 포함률 {result['coverage95']:.3f}, "
            f"대표 건물 {building_id}번, 점검 {'통과' if ok else '확인 필요'}"
        )

    results = pd.DataFrame(rows)
    save_results(results, "bayesian_results.csv")
    save_results(pd.concat(all_predictions, ignore_index=True), "bayesian_valid_predictions.csv")
    plot_r2_bar(results)
    print("\n[유형별 결과]")
    # α, β는 매우 작은 값이라 지수 표기로 출력한다(CSV에는 전체 자릿수로 저장됨).
    print(results.drop(columns=["model"]).to_string(
        index=False,
        formatters={
            "MAE": "{:.1f}".format, "RMSE": "{:.1f}".format, "R2": "{:.4f}".format,
            "coverage95": "{:.3f}".format, "alpha": "{:.3e}".format, "beta": "{:.3e}".format,
        },
    ))


def main():
    if len(sys.argv) > 1:
        building_type = sys.argv[1]
        if building_type not in BUILDING_TYPES:
            raise ValueError(f"건물유형은 {BUILDING_TYPES} 중 하나여야 합니다.")
        run_single(building_type)
    else:
        run_all()


if __name__ == "__main__":
    main()
