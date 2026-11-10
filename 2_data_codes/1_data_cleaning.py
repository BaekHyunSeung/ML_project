# 데이터 정제
# source_data의 훈련·검증 파일에서 결측치, 중복, 이상치를 찾는다.
# 행을 삭제하거나 값을 바꾸지 않고 결과만 출력한다.

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "source_data"

# 시간마다 변하는 값. 강수량은 0이 대부분이라 IQR 기준으로 보지 않는다.
IQR_COLUMNS = [
    "temperature",
    "wind_speed",
    "humidity",
    "sunshine",
    "solar_radiation",
    "power_consumption",
]

# 건물마다 하나씩인 값. 시간 행이 아니라 건물 단위로 확인한다.
BUILDING_COLUMNS = [
    "total_area",
    "cooling_area",
    "solar_capacity",
    "ess_capacity",
    "pcs_capacity",
]


def load_source_files():
    files = []
    for split in ("train", "test"):
        folder = SOURCE_DIR / split
        for path in sorted(folder.glob("*.csv")):
            files.append((split, path))
    return files


def count_missing(data):
    missing = data.isna().sum()
    return missing[missing > 0]


def count_duplicates(data):
    row_duplicates = int(data.duplicated().sum())
    key_duplicates = int(data.duplicated(subset=["num_date_time"]).sum())
    return row_duplicates, key_duplicates


def range_violation_mask(data):
    """값이 물리적으로 들어갈 수 없는 범위를 표시한다."""
    checks = {
        "month": ~data["month"].isin([6, 7, 8]),
        "weekday": ~data["weekday"].between(0, 6),
        "hour": ~data["hour"].between(0, 23),
        "temperature": ~data["temperature"].between(-30, 50),
        "precipitation": data["precipitation"] < 0,
        "wind_speed": data["wind_speed"] < 0,
        "humidity": ~data["humidity"].between(0, 100),
        "total_area": data["total_area"] <= 0,
        "cooling_area": (data["cooling_area"] < 0) | (data["cooling_area"] > data["total_area"]),
        "solar_capacity": data["solar_capacity"] < 0,
        "ess_capacity": data["ess_capacity"] < 0,
        "pcs_capacity": data["pcs_capacity"] < 0,
    }
    if "sunshine" in data.columns:
        checks["sunshine"] = ~data["sunshine"].between(0, 1)
    if "solar_radiation" in data.columns:
        checks["solar_radiation"] = data["solar_radiation"] < 0
    if "power_consumption" in data.columns:
        checks["power_consumption"] = data["power_consumption"] < 0
    return checks


def iqr_outlier_count(values):
    """사분위 범위 바깥 값의 개수를 센다. 범위가 0이면 기준을 만들지 않는다."""
    numbers = values.to_numpy(dtype=np.float64)
    q1, q3 = np.percentile(numbers, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return None
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(np.sum((numbers < lower) | (numbers > upper)))


def building_iqr_outlier_count(data, column):
    building_values = data.groupby("building_id")[column].first()
    return iqr_outlier_count(building_values)


def inspect_file(split, path):
    data = pd.read_csv(path)
    print(f"\n[{split}/{path.name}] {len(data)}행, {data.shape[1]}열")

    missing = count_missing(data)
    if missing.empty:
        print("결측치: 없음")
    else:
        print("결측치:")
        print(missing.to_string())

    row_duplicates, key_duplicates = count_duplicates(data)
    print(f"중복 행: {row_duplicates}")
    print(f"num_date_time 중복: {key_duplicates}")

    print("범위 위반:")
    range_total = 0
    for column, mask in range_violation_mask(data).items():
        count = int(mask.sum())
        range_total += count
        if count:
            print(f"  {column}: {count}")
    if range_total == 0:
        print("  없음")

    print("IQR 이상치(시간 단위):")
    for column in IQR_COLUMNS:
        if column not in data.columns:
            continue
        count = iqr_outlier_count(data[column])
        if count is None:
            print(f"  {column}: 사분위 범위가 0이라 생략")
        else:
            print(f"  {column}: {count}")

    print("IQR 이상치(건물 단위):")
    for column in BUILDING_COLUMNS:
        count = building_iqr_outlier_count(data, column)
        if count is None:
            print(f"  {column}: 사분위 범위가 0이라 생략")
        else:
            print(f"  {column}: {count}")


def main():
    files = load_source_files()
    if not files:
        print(f"파일이 없습니다: {SOURCE_DIR}")
        return
    for split, path in files:
        inspect_file(split, path)


if __name__ == "__main__":
    main()
