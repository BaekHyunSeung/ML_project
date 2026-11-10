# 베이지안 선형 회귀 개선 실험 (기본 결과와 분리)
# 기본 버전(3_bayesian_regression.py)은 팀 입력 명세를 그대로 지켜 다른 모델과 공정하게 비교한다.
# 이 파일은 "입력을 바꾸면 얼마나 좋아지는지"만 확인하는 실험이며, 결과는 별도 CSV로 저장한다.
#
# 실험 목록
#   base         : 기본 명세 그대로 (기본 결과 재현 확인용)
#   +interaction : 시간대 × 주말 상호작용 3열 추가 (주말에는 시간대 패턴이 달라지는 것을 반영)
#   +humidity    : 습도 추가 (습도 0%는 센서 오류로 보고 같은 건물 앞뒤 값으로 선형 보간)
#   +building    : 건물번호 원-핫 추가 (건물마다 다른 기본 사용량 수준을 직접 학습)
#   all          : 위 세 가지 모두

import importlib

import numpy as np
import pandas as pd

from common import BUILDING_TYPES, TARGET, load_train_valid, regression_metrics, save_results

# 파일 이름이 숫자로 시작해서 일반 import 대신 importlib로 불러온다.
bayes = importlib.import_module("3_bayesian_regression")

VARIANTS = {
    "base": [],
    "+interaction": ["interaction"],
    "+humidity": ["humidity"],
    "+building": ["building"],
    "all": ["interaction", "humidity", "building"],
}


# 습도 0% 보간 함수 (IDC 57번 8/5~8/8 등): 0을 결측으로 바꾸고 건물별로 앞뒤 값 사이를 직선으로 채운다.
def fix_humidity(data):
    converted = data.copy()
    humidity = converted["humidity"].replace(0, np.nan)
    converted["humidity"] = humidity.groupby(converted["building_id"]).transform(
        lambda values: values.interpolate(limit_direction="both")
    )
    return converted


def make_variant_features(data, options, building_ids):
    features = bayes.make_features(data)
    continuous = list(bayes.CONTINUOUS_COLUMNS)
    if "interaction" in options:
        for column in bayes.TIME_DUMMY_COLUMNS:
            features[f"weekend_x_{column}"] = features["is_weekend"] * features[column]
    if "humidity" in options:
        features["humidity"] = fix_humidity(data)["humidity"].to_numpy()
        continuous.append("humidity")
    if "building" in options:
        # 첫 번째 건물을 기준으로 두고 나머지 건물만 원-핫 열로 만든다(시간대와 같은 이유).
        for building_id in building_ids[1:]:
            features[f"building_{building_id}"] = (data["building_id"] == building_id).astype(np.float64)
    return features, continuous


def standardize_columns(train_x, valid_x, columns):
    columns = [column for column in columns if column in train_x.columns]
    mean, std = train_x[columns].mean(), train_x[columns].std(ddof=0)
    train_x, valid_x = train_x.copy(), valid_x.copy()
    train_x[columns] = (train_x[columns] - mean) / std
    valid_x[columns] = (valid_x[columns] - mean) / std
    return train_x, valid_x


def run_variant(train, valid, options):
    building_ids = sorted(train["building_id"].unique())
    train_x, continuous = make_variant_features(train, options, building_ids)
    valid_x, _ = make_variant_features(valid, options, building_ids)
    train_x, valid_x, _ = bayes.drop_constant_columns(train_x, valid_x)
    train_x, valid_x = standardize_columns(train_x, valid_x, continuous)

    train_y = train[TARGET].to_numpy(dtype=np.float64)
    valid_y = valid[TARGET].to_numpy(dtype=np.float64)
    alpha, beta, _ = bayes.estimate_alpha_beta(train_x.to_numpy(), train_y)
    x_mean, y_mean = bayes.center(train_x.to_numpy(), train_y)
    m_n, s_n = bayes.fit_posterior(train_x.to_numpy() - x_mean, train_y - y_mean, alpha, beta)
    mean, variance = bayes.predict_distribution(valid_x.to_numpy() - x_mean, m_n, s_n, beta)
    mean = mean + y_mean
    std = np.sqrt(variance)
    return {
        **regression_metrics(valid_y, mean),
        "coverage95": bayes.coverage_95(valid_y, mean, std),
        "n_features": train_x.shape[1],
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
    save_results(table, "bayesian_experiment.csv")

    r2 = table.pivot(index="building_type", columns="variant", values="R2")[list(VARIANTS)]
    mae = table.pivot(index="building_type", columns="variant", values="MAE")[list(VARIANTS)]
    print("\n[실험별 검증 R²]")
    print(r2.round(3).to_string())
    print("\n[실험별 검증 MAE]")
    print(mae.round(1).to_string())
    print("\n[10개 유형 평균]")
    print(table.groupby("variant")[["R2", "MAE", "coverage95"]].mean().loc[list(VARIANTS)].round(3).to_string())


if __name__ == "__main__":
    main()
