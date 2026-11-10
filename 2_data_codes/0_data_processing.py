# 데이터 전처리
# 건물 설비 용량을 숫자로 바꾸고, 건물 정보를 훈련·검증 세트에 붙인 뒤
# 일시를 나누고 건물 유형별 CSV로 저장한다.

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "raw_data"
OUTPUT_DIR = PROJECT_DIR / "source_data"

FACILITY_COLUMNS = ["태양광용량(kW)", "ESS저장용량(kWh)", "PCS용량(kW)"]

BUILDING_COLUMNS = {
    "건물번호": "building_id",
    "건물유형": "building_type",
    "연면적(m2)": "total_area",
    "냉방면적(m2)": "cooling_area",
    "태양광용량(kW)": "solar_capacity",
    "ESS저장용량(kWh)": "ess_capacity",
    "PCS용량(kW)": "pcs_capacity",
}

USAGE_COLUMNS = {
    "num_date_time": "num_date_time",
    "건물번호": "building_id",
    "일시": "date_time",
    "기온(°C)": "temperature",
    "강수량(mm)": "precipitation",
    "풍속(m/s)": "wind_speed",
    "습도(%)": "humidity",
    "일조(hr)": "sunshine",
    "일사(MJ/m2)": "solar_radiation",
    "전력소비량(kWh)": "power_consumption",
}

BUILDING_TYPE_FILES = {
    "백화점": "department_store",
    "호텔": "hotel",
    "상용": "commercial",
    "학교": "school",
    "건물기타": "other",
    "병원": "hospital",
    "아파트": "apartment",
    "연구소": "research",
    "IDC(전화국)": "idc",
    "공공": "public",
}


def load_csv(filename):
    return pd.read_csv(DATA_DIR / filename)

# 설비 시설 용량 자료형 변환 함수
def convert_facility_capacity(building):
    converted = building.copy()
    for column in FACILITY_COLUMNS:
        values = converted[column].replace("-", "0").to_numpy(dtype=str)
        converted[column] = values.astype(np.float64)
    return converted


# 일시 변환 함수
def split_date_time(data):
    converted = data.copy()
    date_time = pd.to_datetime(converted["date_time"], format="%Y%m%d %H")
    converted["month"] = date_time.dt.month.to_numpy()
    converted["weekday"] = date_time.dt.dayofweek.to_numpy()
    converted["hour"] = date_time.dt.hour.to_numpy()
    return converted.drop(columns=["date_time"])


# 건물 정보 추가 함수
def attach_building_info(usage, building):
    building_columns = [
        "building_id",
        "building_type",
        "total_area",
        "cooling_area",
        "solar_capacity",
        "ess_capacity",
        "pcs_capacity",
    ]
    return usage.merge(building[building_columns], on="building_id", how="left")


# 건물 유형별 CSV 파일 저장 함수
def save_by_building_type(data, folder_name):
    output_folder = OUTPUT_DIR / folder_name
    output_folder.mkdir(parents=True, exist_ok=True)
    for building_type, group in data.groupby("building_type", sort=True):
        file_name = BUILDING_TYPE_FILES[building_type]
        group.to_csv(output_folder / f"{file_name}.csv", index=False)
        print(f"{folder_name}/{file_name}.csv: {len(group)}행")


def main():
    building = convert_facility_capacity(load_csv("building_info.csv"))
    building = building.rename(columns=BUILDING_COLUMNS)

    train = attach_building_info(
        split_date_time(load_csv("train.csv").rename(columns=USAGE_COLUMNS)),
        building,
    )
    test = attach_building_info(
        split_date_time(load_csv("test.csv").rename(columns=USAGE_COLUMNS)),
        building,
    )

    save_by_building_type(train, "train")
    save_by_building_type(test, "test")


if __name__ == "__main__":
    main()
