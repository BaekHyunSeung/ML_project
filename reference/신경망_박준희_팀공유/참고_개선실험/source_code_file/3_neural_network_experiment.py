# 인공신경망 개선 실험 (기본 결과와 분리)
# 기본 버전(3_neural_network.py)은 팀 입력 명세를 그대로 지켜 다른 모델과 공정하게 비교한다.
# 이 파일은 "입력을 바꾸면 얼마나 달라지는지"만 확인하는 실험이며, 결과는 별도 CSV로 저장한다.
# 신경망 구조·학습 설정(은닉층, 조기 종료 등)은 기본 버전과 똑같이 두고 입력만 바꾼다.
#
# 실험 목록
#   base           : 기본 명세 그대로 (기본 결과 재현 확인용)
#   -building_info : 면적·설비 5개 열을 뺀다 → 신경망이 이 열들로 "건물을 구분"하고 있는지 확인
#   +building      : 건물번호 원-핫 추가 (건물마다 다른 사용량 수준을 직접 학습)
#   +weather_time  : 습도·풍속·강수량, 시간·요일 주기형(sin/cos), 공휴일 추가
#                    (4구간 시간대는 계단 모양 예측이 되므로 24시간을 매끄럽게 표현)
#   all            : +building + +weather_time

import importlib

import numpy as np
import pandas as pd

from common import (
    BUILDING_TYPES,
    TARGET,
    drop_constant_columns,
    fix_humidity_zero,
    load_train_valid,
    make_base_features,
    regression_metrics,
    save_results,
    standardize,
)

# 파일 이름이 숫자로 시작해서 일반 import 대신 importlib로 불러온다.
nn = importlib.import_module("3_neural_network")

BUILDING_INFO_COLUMNS = ["total_area", "cooling_area", "solar_capacity", "ess_capacity", "pcs_capacity"]
WEATHER_COLUMNS = ["humidity", "wind_speed", "precipitation"]
# 학습 기간 공휴일: 6/6 현충일, 8/15 광복절(검증 기간). test 기간(8/25~31)에는 공휴일 없음
HOLIDAYS = pd.to_datetime(["2024-06-06", "2024-08-15"])

VARIANTS = {
    "base": [],
    "-building_info": ["no_building_info"],
    "+building": ["building"],
    "+weather_time": ["weather_time"],
    "all": ["building", "weather_time"],
}


# 주기형 시간 함수: 23시와 0시, 일요일과 월요일이 서로 가깝게 이어지도록 원 위의 좌표(sin, cos)로 바꾼다.
def add_cyclic_time(features, data):
    hour = data["hour"].to_numpy()
    weekday = data["weekday"].to_numpy()
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    return features


# 공휴일 함수: 주말과 공휴일을 합쳐 '쉬는 날'로 표시한다.
def add_off_day(features, data):
    is_holiday = data["date_time"].dt.normalize().isin(HOLIDAYS).to_numpy()
    features["is_off_day"] = ((features["is_weekend"].to_numpy() == 1) | is_holiday).astype(np.float64)
    return features


def make_variant_features(data, options, building_ids):
    features = make_base_features(data)
    continuous = ["temperature", "month"] + BUILDING_INFO_COLUMNS
    if "no_building_info" in options:
        features = features.drop(columns=BUILDING_INFO_COLUMNS)
    if "weather_time" in options:
        fixed = fix_humidity_zero(data)
        for column in WEATHER_COLUMNS:
            features[column] = fixed[column].to_numpy()
        continuous += WEATHER_COLUMNS
        features = add_cyclic_time(features, data)
        features = add_off_day(features, data)
    if "building" in options:
        # 첫 번째 건물을 기준으로 두고 나머지 건물만 원-핫 열로 만든다.
        for building_id in building_ids[1:]:
            features[f"building_{building_id}"] = (data["building_id"] == building_id).astype(np.float64)
    return features, continuous


def run_variant(train, valid, options):
    building_ids = sorted(train["building_id"].unique())
    train_x, continuous = make_variant_features(train, options, building_ids)
    valid_x, _ = make_variant_features(valid, options, building_ids)
    train_x, valid_x, _ = drop_constant_columns(train_x, valid_x)
    train_x, valid_x = standardize(train_x, valid_x, continuous)

    train_y = train[TARGET].to_numpy(dtype=np.float64)
    valid_y = valid[TARGET].to_numpy(dtype=np.float64)
    train_y_scaled, y_mean, y_std = nn.scale_target(train_y)
    model, converged = nn.train_model(train_x.to_numpy(), train_y_scaled)
    pred = nn.predict(model, valid_x.to_numpy(), y_mean, y_std)
    return {
        **regression_metrics(valid_y, pred),
        "n_features": train_x.shape[1],
        "n_epochs": model.n_iter_,
        "converged": converged,
    }


def main():
    rows = []
    for building_type in BUILDING_TYPES:
        train, valid = load_train_valid(building_type, verbose=False)
        for name, options in VARIANTS.items():
            result = run_variant(train, valid, options)
            rows.append({"building_type": building_type, "variant": name, **result})
        print(f"{building_type} 완료")

    table = pd.DataFrame(rows)
    save_results(table, "neural_network_experiment.csv")

    r2 = table.pivot(index="building_type", columns="variant", values="R2")[list(VARIANTS)]
    mae = table.pivot(index="building_type", columns="variant", values="MAE")[list(VARIANTS)]
    print("\n[실험별 검증 R²]")
    print(r2.round(3).to_string())
    print("\n[실험별 검증 MAE]")
    print(mae.round(1).to_string())
    print("\n[10개 유형 평균]")
    print(table.groupby("variant")[["R2", "MAE"]].mean().loc[list(VARIANTS)].round(3).to_string())
    print(f"\n수렴하지 않은 실행: {int((~table['converged']).sum())}개")


if __name__ == "__main__":
    main()
