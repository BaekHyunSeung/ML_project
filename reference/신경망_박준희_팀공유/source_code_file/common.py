# 팀 공용 모듈
# 모든 모델 담당이 같은 방식으로 데이터를 읽고, 나누고, 평가하도록 공통 함수를 모아 둔다.
# 사용 예:
#   from common import load_train_valid, regression_metrics, save_results
#   train, valid = load_train_valid("public")

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "source_data"
RESULTS_DIR = PROJECT_DIR / "results"

TARGET = "power_consumption"

# 0_data_processing.py가 만든 파일 이름과 같은 순서(알파벳순)로 둔다.
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

# 그래프 제목 등에 쓸 한글 이름
BUILDING_TYPE_NAMES = {
    "apartment": "아파트",
    "commercial": "상용",
    "department_store": "백화점",
    "hospital": "병원",
    "hotel": "호텔",
    "idc": "IDC(전화국)",
    "other": "건물기타",
    "public": "공공",
    "research": "연구소",
    "school": "학교",
}

# 날짜 기준 분할 시점: 이 시각 미만은 학습(6/1~8/7, 68일), 이후는 검증(8/8~8/24, 17일).
# 행 순서로 80%를 자르면 "기간"이 아니라 "건물"이 잘리므로(호텔은 10동 중 2동이 검증으로 감)
# 날짜로 잘라야 모든 건물이 학습·검증에 함께 들어간다. 10개 유형 모두 정확히 80:20이 된다.
SPLIT_DATE = pd.Timestamp("2024-08-08 00:00")


# 건물유형 CSV 읽기 함수 (split: "train" 또는 "test")
def load_type_data(building_type, split="train"):
    path = SOURCE_DIR / split / f"{building_type}.csv"
    if not path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {path}")
    return pd.read_csv(path)


# 날짜 생성 함수
# 전처리 파일에는 date_time 열이 없으므로 num_date_time("건물번호_YYYYMMDD HH")에서 꺼낸다.
def add_datetime(data):
    converted = data.copy()
    date_text = converted["num_date_time"].str.split("_").str[1]
    converted["date_time"] = pd.to_datetime(date_text, format="%Y%m%d %H")
    return converted


# 날짜 기준 학습/검증 분할 함수
def split_train_valid(data, split_date=SPLIT_DATE):
    is_train = data["date_time"] < split_date
    train = data[is_train].reset_index(drop=True)
    valid = data[~is_train].reset_index(drop=True)
    return train, valid


# 전력소비량 0인 행 제외 함수
# 0은 몇 시간씩 연속으로 나타나서 정전·계측 누락으로 판단한다(전체 68행).
# 학습에 남기면 모델이 잘못 배우고, 검증에 남기면 실제로 맞힐 수 없는 값으로 점수가 깎인다.
def remove_zero_power(data):
    is_zero = data[TARGET] == 0
    return data[~is_zero].reset_index(drop=True), int(is_zero.sum())


# 학습·검증 세트 준비 함수 (로드 → 날짜 생성 → 분할 → 0값 제외)
def load_train_valid(building_type, verbose=True):
    data = add_datetime(load_type_data(building_type, "train"))
    train, valid = split_train_valid(data)
    train, train_zero = remove_zero_power(train)
    valid, valid_zero = remove_zero_power(valid)
    if verbose:
        print(
            f"[{building_type}] 학습 {len(train)}행(0값 {train_zero}행 제외), "
            f"검증 {len(valid)}행(0값 {valid_zero}행 제외), 건물 {data['building_id'].nunique()}동"
        )
    return train, valid


# test.csv 읽기 함수 (정답이 없으므로 최종 예측값 생성용으로만 쓴다)
def load_test(building_type):
    return add_datetime(load_type_data(building_type, "test"))


# ---------------------------------------------------------------
# 팀 입력 명세 (기본 버전) — 모든 모델이 같은 입력으로 공정하게 비교하기 위한 함수
# 팀 전처리 문서 기준: 기온, 월, 시간대(4구간), 주말 여부, 면적·설비 용량 5개
# 일조·일사는 test.csv에 없어서 제외한다.
# ---------------------------------------------------------------

BASE_CONTINUOUS_COLUMNS = [
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
BASE_BINARY_COLUMNS = ["is_weekend"] + TIME_DUMMY_COLUMNS


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


# 팀 입력 명세 입력 행렬 생성 함수
def make_base_features(data):
    converted = add_weekend(add_time_dummies(data))
    return converted[BASE_CONTINUOUS_COLUMNS + BASE_BINARY_COLUMNS].astype(np.float64)


# 값이 모두 같은 열 제외 함수
# 아파트는 설비 3개 열, IDC·상용·호텔은 ESS·PCS가 전부 0이다.
# 표준편차가 0이면 표준화할 때 0으로 나누게 되고, 정보도 없으므로 학습 데이터 기준으로 뺀다.
def drop_constant_columns(train_x, valid_x):
    constant = [column for column in train_x.columns if train_x[column].nunique() <= 1]
    return train_x.drop(columns=constant), valid_x.drop(columns=constant), constant


# 표준화 함수 (columns를 안 주면 팀 명세의 연속형 열만 표준화, 0/1 열은 그대로 둔다)
# 평균·표준편차는 학습 데이터에서만 구하고 검증 데이터에는 같은 값을 적용만 한다.
# 검증 데이터의 통계를 쓰면 미래 정보를 미리 보는 셈(데이터 누수)이 된다.
def standardize(train_x, valid_x, columns=None):
    if columns is None:
        columns = BASE_CONTINUOUS_COLUMNS
    columns = [column for column in columns if column in train_x.columns]
    mean = train_x[columns].mean()
    std = train_x[columns].std(ddof=0)
    train_scaled, valid_scaled = train_x.copy(), valid_x.copy()
    train_scaled[columns] = (train_x[columns] - mean) / std
    valid_scaled[columns] = (valid_x[columns] - mean) / std
    return train_scaled, valid_scaled


# 습도 0% 보간 함수 (습도를 쓰는 모델용)
# 습도 0%는 센서 오류로 본다(학습: IDC 57번 8/5~8/8, test: 호텔 28번 8/29).
# 0을 결측으로 바꾸고 같은 건물의 앞뒤 값 사이를 직선으로 채운다.
def fix_humidity_zero(data):
    converted = data.copy()
    humidity = converted["humidity"].replace(0, np.nan)
    converted["humidity"] = humidity.groupby(converted["building_id"]).transform(
        lambda values: values.interpolate(limit_direction="both")
    )
    return converted


# 그래프 대표 건물 선택 함수: 검증 기간 평균 사용량이 유형 안에서 중앙값에 가장 가까운 건물
# (모든 모델이 같은 건물을 그려야 그래프끼리 비교할 수 있다)
def pick_representative_building(predictions, actual_column="actual"):
    building_mean = predictions.groupby("building_id")[actual_column].mean()
    median = building_mean.median()
    return int((building_mean - median).abs().idxmin())


# 팀 공통 평가지표 함수: MAE, RMSE, R²
def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    error = y_true - y_pred
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    # R² = 1 - (모델 오차 제곱합 / 평균으로 예측했을 때의 오차 제곱합)
    # 0이면 "평균만 찍는 것"과 같은 수준, 1에 가까울수록 좋다.
    r2 = 1 - np.sum(error ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# 결과 저장 함수 (dict 목록 또는 DataFrame을 results 폴더에 CSV로 저장)
def save_results(rows, filename):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    path = RESULTS_DIR / filename
    # 엑셀에서 한글이 깨지지 않도록 utf-8-sig로 저장한다.
    table.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"저장: {path}")
    return path


# 그래프 한글 폰트 설정 함수
def setup_korean_font():
    import matplotlib.pyplot as plt

    # Windows는 Malgun Gothic, 없는 환경에서는 Noto Sans CJK KR로 대체한다.
    plt.rcParams["font.family"] = ["Malgun Gothic", "Noto Sans CJK KR", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


if __name__ == "__main__":
    # 공용 모듈 점검: 10개 유형 분할 비율과 0값 제외 행 수를 출력한다.
    total_zero = 0
    for building_type in BUILDING_TYPES:
        data = add_datetime(load_type_data(building_type, "train"))
        train, valid = split_train_valid(data)
        ratio = len(train) / len(data)
        zero = int((data[TARGET] == 0).sum())
        total_zero += zero
        print(
            f"{building_type:17s} 전체 {len(data):6d}  학습 {len(train):6d}  검증 {len(valid):5d}  "
            f"학습비율 {ratio:.4f}  학습일수 {train['date_time'].dt.date.nunique()}  "
            f"검증일수 {valid['date_time'].dt.date.nunique()}  0값 {zero}"
        )
    print(f"전력소비량 0인 행 합계: {total_zero}")
