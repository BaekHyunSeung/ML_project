# 인공신경망 (다층 퍼셉트론, 담당: 박준희)
# 건물유형마다 따로 학습하고, 날짜 기준 검증 20%로 평가한다(베이지안과 같은 기준).
# 입력은 common.py의 팀 입력 명세를 그대로 써서 다른 모델과 공정하게 비교한다.
#
# 신경망 한 줄 요약
#   입력(기온, 시간대 등) → 은닉층 64개 노드 → 은닉층 32개 노드 → 출력(전력소비량)
#   각 노드는 "가중합 + ReLU(음수는 0으로)"를 계산하고, 오차가 줄어드는 방향으로 가중치를 조금씩 고친다(역전파).
#   선형 회귀는 입력마다 기울기 하나만 배우지만, 신경망은 ReLU를 거치며 "기온이 높고 + 오후이고 + 큰 건물일 때"
#   같은 입력 사이의 조합(비선형 관계)을 스스로 만든다.
#
# 실행 방법
#   python 3_neural_network.py public   → 공공 유형 하나만 실행하고 정상 작동을 점검
#   python 3_neural_network.py          → 10개 유형 전체 실행 + 결과 CSV + 그래프 저장

import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

from common import (
    BUILDING_TYPE_NAMES,
    BUILDING_TYPES,
    RESULTS_DIR,
    TARGET,
    drop_constant_columns,
    load_train_valid,
    make_base_features,
    pick_representative_building,
    regression_metrics,
    save_results,
    setup_korean_font,
    standardize,
)

MODEL_NAME = "neural_network"

# 하이퍼파라미터 (검증 점수를 보고 고르지 않고 미리 정한 값. 이유는 주석과 DECISIONS.md 참고)
HIDDEN_LAYERS = (64, 32)  # 은닉층 2개. 입력이 10개 안팎이라 이 정도면 충분하다(크기를 바꿔도 결과 차이가 작았음)
L2_ALPHA = 1e-3           # 가중치가 너무 커지지 않게 하는 L2 규제 세기 (sklearn 이름이 alpha라 베이지안 α와 헷갈리지 말 것)
LEARNING_RATE = 1e-3      # Adam 옵티마이저의 학습률 (sklearn 기본값)
MAX_EPOCHS = 500          # 최대 반복 횟수(에폭). 보통 조기 종료로 그 전에 멈춘다
PATIENCE = 20             # 내부 검증 점수가 20 에폭 연속 좋아지지 않으면 멈춘다
EARLY_STOP_FRACTION = 0.1 # 조기 종료 판단용으로 학습 데이터 안에서 떼어 두는 비율 (우리 검증 20%와는 별개)
RANDOM_STATE = 42         # 초기 가중치를 정하는 난수. 고정해야 매번 같은 결과가 나온다
CHECK_SEEDS = [42, 7, 2026]  # 정상 작동 점검 때 난수를 바꿔 결과가 흔들리는지 본다


# ---------------------------------------------------------------
# 1. 입력·출력 준비
# ---------------------------------------------------------------

# 입력 준비 함수 (베이지안과 같은 순서: 팀 명세 입력 → 상수 열 제외 → 학습 통계로 표준화)
# 신경망은 입력 크기가 제각각이면(면적 수만 vs 0/1) 큰 값 쪽 가중치만 크게 움직여 학습이 불안정해진다.
# 그래서 표준화가 선형 모델보다 더 중요하다.
def prepare_inputs(train, valid):
    train_x, valid_x = make_base_features(train), make_base_features(valid)
    train_x, valid_x, dropped = drop_constant_columns(train_x, valid_x)
    train_x, valid_x = standardize(train_x, valid_x)
    return train_x, valid_x, dropped


# 출력 표준화 함수
# 전력소비량은 수천~수만 kWh라 그대로 학습하면 초기 오차가 너무 커서 가중치가 크게 튄다.
# 학습 데이터의 평균·표준편차로 평균 0, 표준편차 1로 바꿔 학습하고, 예측 후 원래 단위로 되돌린다.
def scale_target(train_y):
    mean, std = train_y.mean(), train_y.std()
    return (train_y - mean) / std, mean, std


def unscale_target(scaled, mean, std):
    return scaled * std + mean


# ---------------------------------------------------------------
# 2. 신경망 학습·예측
# ---------------------------------------------------------------

# 모델 생성 함수
# early_stopping=True: 학습 데이터 안에서 10%를 떼어 두고, 그 점수가 더 안 좋아지면 학습을 멈춘다.
# 너무 오래 학습해서 학습 데이터만 외우는 것(과적합)을 막기 위함이다.
# 우리가 평가에 쓰는 검증 20%(8/8 이후)는 학습에 전혀 쓰지 않는다.
def build_model(random_state=RANDOM_STATE):
    return MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        alpha=L2_ALPHA,
        learning_rate_init=LEARNING_RATE,
        max_iter=MAX_EPOCHS,
        early_stopping=True,
        validation_fraction=EARLY_STOP_FRACTION,
        n_iter_no_change=PATIENCE,
        random_state=random_state,
    )


# 학습 함수 (최대 에폭까지 가서 멈추면 '수렴 안 됨' 경고가 뜨는데, 그 여부를 기록해 점검에 쓴다)
def train_model(train_x, train_y_scaled, random_state=RANDOM_STATE):
    model = build_model(random_state)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train_x, train_y_scaled)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, converged


def predict(model, valid_x, mean, std):
    return unscale_target(model.predict(valid_x), mean, std)


# ---------------------------------------------------------------
# 3. 유형 하나 학습·검증
# ---------------------------------------------------------------

def run_one_type(building_type, verbose=True, random_state=RANDOM_STATE):
    train, valid = load_train_valid(building_type, verbose=verbose)
    train_x, valid_x, dropped = prepare_inputs(train, valid)
    train_y = train[TARGET].to_numpy(dtype=np.float64)
    valid_y = valid[TARGET].to_numpy(dtype=np.float64)
    if verbose and dropped:
        print(f"  값이 모두 같아 제외한 열: {dropped}")

    train_y_scaled, y_mean, y_std = scale_target(train_y)
    model, converged = train_model(train_x.to_numpy(), train_y_scaled, random_state)
    pred = predict(model, valid_x.to_numpy(), y_mean, y_std)

    result = {
        "model": MODEL_NAME,
        "building_type": building_type,
        "n_train": len(train),
        "n_valid": len(valid),
        **regression_metrics(valid_y, pred),
        "n_epochs": model.n_iter_,
        "converged": converged,
        "hidden_layers": "-".join(str(n) for n in HIDDEN_LAYERS),
    }
    predictions = pd.DataFrame(
        {
            "building_type": building_type,
            "building_id": valid["building_id"].to_numpy(),
            "date_time": valid["date_time"].to_numpy(),
            "actual": valid_y,
            "pred": pred,
        }
    )
    extras = {"model": model, "train_y": train_y}
    return result, predictions, extras


# ---------------------------------------------------------------
# 4. 정상 작동 점검 (유형 1개로 먼저 확인)
# ---------------------------------------------------------------

# 같은 유형의 베이지안 결과를 읽는 함수 (results/bayesian_results.csv가 있을 때만)
def load_bayesian_r2(building_type):
    path = RESULTS_DIR / "bayesian_results.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    row = table[table["building_type"] == building_type]
    return None if row.empty else float(row["R2"].iloc[0])


# 난수(초기 가중치)를 바꿔도 결과가 비슷한지 확인하는 함수
# 신경망은 시작점에 따라 다른 답에 도착할 수 있어서, 결과가 크게 흔들리면 믿기 어렵다.
def seed_stability(building_type):
    scores = []
    for seed in CHECK_SEEDS:
        result, _, _ = run_one_type(building_type, verbose=False, random_state=seed)
        scores.append(result["R2"])
    return np.array(scores)


def check_one_type(result, predictions, extras, verbose=True):
    pred = predictions["pred"].to_numpy()
    actual = predictions["actual"].to_numpy()

    no_nan = not np.isnan(pred).any()
    same_count = len(pred) == result["n_valid"]

    # 단순 평균 예측: 학습 데이터 평균 하나로 모든 시간을 예측했을 때
    baseline = np.full_like(actual, extras["train_y"].mean())
    baseline_metrics = regression_metrics(actual, baseline)
    beats_baseline = result["RMSE"] < baseline_metrics["RMSE"]

    # 조기 종료로 멈췄는지 (최대 에폭까지 갔으면 학습이 덜 끝났을 수 있다)
    stopped_early = result["converged"] and result["n_epochs"] < MAX_EPOCHS

    ok = all([no_nan, same_count, beats_baseline, stopped_early])
    if not verbose:
        return ok

    print("\n[정상 작동 점검]")
    print(f"  NaN 없음: {no_nan}")
    print(f"  예측 개수 = 검증 행 수: {same_count} ({len(pred)} / {result['n_valid']})")
    print(f"  지표: MAE {result['MAE']:.2f}, RMSE {result['RMSE']:.2f}, R² {result['R2']:.4f}")
    print(
        f"  단순 평균 예측: MAE {baseline_metrics['MAE']:.2f}, RMSE {baseline_metrics['RMSE']:.2f}, "
        f"R² {baseline_metrics['R2']:.4f} → 신경망이 더 나음: {beats_baseline}"
    )
    print(f"  조기 종료로 학습 종료: {stopped_early} ({result['n_epochs']} 에폭 / 최대 {MAX_EPOCHS})")

    scores = seed_stability(result["building_type"])
    stable = (scores.max() - scores.min()) < 0.05
    print(
        f"  난수 {CHECK_SEEDS}별 R²: {np.round(scores, 4).tolist()} → 차이 {scores.max() - scores.min():.4f}, "
        f"안정적(차이 0.05 미만): {stable}"
    )

    bayes_r2 = load_bayesian_r2(result["building_type"])
    if bayes_r2 is not None:
        print(f"  같은 입력·같은 분할의 베이지안 R² {bayes_r2:.4f} → 신경망 R² {result['R2']:.4f}")
    return ok and stable


# ---------------------------------------------------------------
# 5. 시각화
# ---------------------------------------------------------------

def plot_prediction(predictions, building_type):
    # 대표 건물은 베이지안 그래프와 같은 기준(common.pick_representative_building)으로 골라서 서로 비교할 수 있다.
    building_id = pick_representative_building(predictions)
    part = predictions[predictions["building_id"] == building_id].sort_values("date_time")

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(part["date_time"], part["actual"], color="black", linewidth=1, label="실제값")
    ax.plot(part["date_time"], part["pred"], color="tab:orange", linewidth=1.2, label="신경망 예측")
    name = BUILDING_TYPE_NAMES[building_type]
    ax.set_title(f"{name} - {building_id}번 건물 검증 기간 예측 (인공신경망)")
    ax.set_xlabel("일시")
    ax.set_ylabel("전력소비량(kWh)")
    ax.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"neural_network_pred_{building_type}.png", dpi=120)
    plt.close(fig)
    return building_id


def plot_r2_bar(results):
    table = results.sort_values("R2", ascending=False)
    labels = [BUILDING_TYPE_NAMES[t] for t in table["building_type"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, table["R2"], color="tab:orange")
    ax.bar_label(bars, fmt="%.2f", padding=2)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylim(min(0, table["R2"].min()) - 0.05, 1.08)
    ax.set_title("건물유형별 검증 R² (인공신경망)")
    ax.set_ylabel("R²")
    ax.set_xlabel("건물유형")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "neural_network_r2_by_type.png", dpi=120)
    plt.close(fig)


# 베이지안과 신경망 R² 비교 그래프 (같은 입력·같은 분할이라 바로 비교할 수 있다)
def plot_compare_with_bayesian(results):
    path = RESULTS_DIR / "bayesian_results.csv"
    if not path.exists():
        print("bayesian_results.csv가 없어 비교 그래프를 건너뜁니다.")
        return
    bayes = pd.read_csv(path)[["building_type", "R2"]].rename(columns={"R2": "bayesian"})
    table = results[["building_type", "R2"]].rename(columns={"R2": "neural_network"})
    table = table.merge(bayes, on="building_type").sort_values("neural_network", ascending=False)
    labels = [BUILDING_TYPE_NAMES[t] for t in table["building_type"]]
    x = np.arange(len(table))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, table["bayesian"], width=0.4, color="tab:blue", label="베이지안 선형 회귀")
    ax.bar(x + 0.2, table["neural_network"], width=0.4, color="tab:orange", label="인공신경망")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylim(0, 1.08)
    ax.set_title("건물유형별 검증 R² 비교 (같은 입력, 같은 분할)")
    ax.set_ylabel("R²")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "neural_network_vs_bayesian_r2.png", dpi=120)
    plt.close(fig)


# 학습 곡선 그래프: 에폭마다 학습 오차(loss)가 줄어드는 모습. 평평해지면 학습이 끝난 것.
def plot_loss_curves(loss_curves):
    fig, ax = plt.subplots(figsize=(10, 5))
    for building_type, curve in loss_curves.items():
        ax.plot(np.arange(1, len(curve) + 1), curve, linewidth=1.2, label=BUILDING_TYPE_NAMES[building_type])
    ax.set_yscale("log")
    ax.set_title("건물유형별 학습 곡선 (표준화된 출력 기준 학습 오차)")
    ax.set_xlabel("에폭")
    ax.set_ylabel("학습 오차 (log 눈금)")
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "neural_network_loss_curves.png", dpi=120)
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
    rows, all_predictions, loss_curves = [], [], {}
    for building_type in BUILDING_TYPES:
        result, predictions, extras = run_one_type(building_type)
        # 10개 유형 모두 기본 점검(NaN, 개수, 평균 예측 대비, 조기 종료)을 조용히 수행한다.
        ok = check_one_type(result, predictions, extras, verbose=False)
        rows.append(result)
        all_predictions.append(predictions)
        loss_curves[building_type] = extras["model"].loss_curve_
        building_id = plot_prediction(predictions, building_type)
        print(
            f"  → R² {result['R2']:.3f}, MAE {result['MAE']:.1f}, {result['n_epochs']} 에폭, "
            f"대표 건물 {building_id}번, 점검 {'통과' if ok else '확인 필요'}"
        )

    results = pd.DataFrame(rows)
    save_results(results, "neural_network_results.csv")
    save_results(pd.concat(all_predictions, ignore_index=True), "neural_network_valid_predictions.csv")
    plot_r2_bar(results)
    plot_compare_with_bayesian(results)
    plot_loss_curves(loss_curves)
    print("\n[유형별 결과]")
    print(results.drop(columns=["model", "hidden_layers"]).to_string(
        index=False,
        formatters={"MAE": "{:.1f}".format, "RMSE": "{:.1f}".format, "R2": "{:.4f}".format},
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
