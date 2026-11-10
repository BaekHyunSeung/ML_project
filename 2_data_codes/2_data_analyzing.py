# 데이터 분석
# 훈련 세트의 전력소비량이 건물유형, 시간, 면적, 날씨와 어떻게 연결되는지 본다.
# 값을 바꾸지 않고, 표는 화면에 출력하고 그래프는 analysis 폴더에 저장한다.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_DIR / "source_data" / "train"
OUTPUT_DIR = PROJECT_DIR / "analysis"

# 모델 입력으로 쓸 열과, 훈련에만 있는 참고 열.
FEATURE_COLUMNS = [
    "temperature",
    "humidity",
    "wind_speed",
    "hour",
    "weekday",
    "month",
    "total_area",
    "cooling_area",
    "solar_capacity",
    "ess_capacity",
    "pcs_capacity",
    "sunshine",
    "solar_radiation",
]
TARGET = "power_consumption"


def load_train():
    frames = [pd.read_csv(path) for path in sorted(TRAIN_DIR.glob("*.csv"))]
    if not frames:
        raise FileNotFoundError(f"훈련 파일이 없습니다: {TRAIN_DIR}")
    return pd.concat(frames, ignore_index=True)


def setup_plot():
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def print_power_summary(data):
    summary = (
        data.groupby("building_type")[TARGET]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(1)
        .sort_values("mean", ascending=False)
    )
    print("\n[건물유형별 전력소비량]")
    print(summary.to_string())


def print_time_summary(data):
    by_hour = data.groupby("hour")[TARGET].mean().round(1)
    by_weekday = data.groupby("weekday")[TARGET].mean().round(1)
    by_month = data.groupby("month")[TARGET].mean().round(1)
    print("\n[시간별 평균 전력소비량]")
    print(by_hour.to_string())
    print("\n[요일별 평균 전력소비량] 0=월 ... 6=일")
    print(by_weekday.to_string())
    print("\n[월별 평균 전력소비량]")
    print(by_month.to_string())


def correlation_table(data):
    columns = FEATURE_COLUMNS + [TARGET]
    return data[columns].corr(numeric_only=True)[TARGET].drop(TARGET)


def print_correlations(data):
    overall = correlation_table(data).sort_values(key=np.abs, ascending=False).round(3)
    print("\n[전체 데이터와 전력소비량의 상관]")
    print(overall.to_string())

    print("\n[건물유형 안에서 전력소비량과의 상관]")
    rows = []
    for building_type, group in data.groupby("building_type"):
        row = correlation_table(group).round(3)
        row["building_type"] = building_type
        rows.append(row)
    by_type = pd.DataFrame(rows).set_index("building_type")
    print(by_type.to_string())
    return by_type


def building_level(data):
    """면적은 건물마다 같으므로 건물 평균 사용량과 비교한다."""
    return (
        data.groupby(["building_type", "building_id"], as_index=False)
        .agg(
            power=(TARGET, "mean"),
            total_area=("total_area", "first"),
            cooling_area=("cooling_area", "first"),
        )
    )


def print_area_correlation(buildings):
    print("\n[건물 평균 사용량과 면적의 상관]")
    for building_type, group in buildings.groupby("building_type"):
        area_corr = group["power"].corr(group["total_area"])
        cooling_corr = group["power"].corr(group["cooling_area"])
        print(
            f"{building_type}: 연면적 {area_corr:+.2f}, 냉방면적 {cooling_corr:+.2f}, 건물 {len(group)}동"
        )


def save_power_by_type(data):
    order = data.groupby("building_type")[TARGET].mean().sort_values(ascending=False).index
    plt.figure(figsize=(10, 5))
    sns.barplot(data=data, x="building_type", y=TARGET, order=order, errorbar=None)
    plt.title("건물유형별 평균 전력소비량")
    plt.xlabel("건물유형")
    plt.ylabel("전력소비량(kWh)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "power_by_type.png", dpi=120)
    plt.close()


def save_power_by_hour(data):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="hour", y=TARGET, hue="building_type", errorbar=None)
    plt.title("시간별 평균 전력소비량")
    plt.xlabel("시")
    plt.ylabel("전력소비량(kWh)")
    plt.legend(title="건물유형", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "power_by_hour.png", dpi=120)
    plt.close()


def save_power_by_weekday(data):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x="weekday", y=TARGET, hue="building_type", errorbar=None)
    plt.title("요일별 평균 전력소비량")
    plt.xlabel("요일 (0=월 ... 6=일)")
    plt.ylabel("전력소비량(kWh)")
    plt.legend(title="건물유형", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "power_by_weekday.png", dpi=120)
    plt.close()


def save_correlation_heatmap(data):
    columns = FEATURE_COLUMNS + [TARGET]
    corr = data[columns].corr(numeric_only=True)
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("입력 변수와 전력소비량의 상관")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=120)
    plt.close()


def save_area_scatter(buildings):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=buildings, x="total_area", y="power", hue="building_type")
    plt.title("건물 연면적과 평균 전력소비량")
    plt.xlabel("연면적(m2)")
    plt.ylabel("평균 전력소비량(kWh)")
    plt.legend(title="건물유형", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "area_vs_power.png", dpi=120)
    plt.close()


def main():
    setup_plot()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_train()
    print(f"훈련 데이터: {len(data)}행, {data['building_id'].nunique()}동")

    print_power_summary(data)
    print_time_summary(data)
    print_correlations(data)
    buildings = building_level(data)
    print_area_correlation(buildings)

    save_power_by_type(data)
    save_power_by_hour(data)
    save_power_by_weekday(data)
    save_correlation_heatmap(data)
    save_area_scatter(buildings)
    print(f"\n그래프 저장: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
