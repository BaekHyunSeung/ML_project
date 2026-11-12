# 팀 공용 모듈
# 모든 모델 담당이 같은 방식으로 데이터를 읽고, 나누고, 평가하도록 공통 함수를 모아 둔다.
# 사용 예:
#   from common import load_train_valid, regression_metrics, save_results
#   train, valid = load_train_valid("public")

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "1_source_data"
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

# 학습과 검증은 날짜가 아니라 건물로 나눈다.
# 연면적이 작은 순으로 1/3, 2/3 위치의 2동을 검증으로 뺀다.
# 그 2동은 전체 기간으로 점수를 내고, 나머지 건물은 전체 기간을 학습에 쓴다.
# 가장 작은 건물과 가장 큰 건물은 학습에 남긴다. 한 번도 보지 못한 크기를 맞히는 검증이 되지 않게 한다.


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


def holdout_building_ids(data):
    """연면적 오름차순에서 1/3, 2/3 위치의 건물 2동을 검증용으로 고른다."""
    area = (
        data.groupby("building_id", sort=False)["total_area"]
        .first()
        .reset_index()
        .sort_values(["total_area", "building_id"], kind="mergesort")
    )
    count = len(area)
    if count < 3:
        raise ValueError("검증으로 빼려면 건물이 3동 이상이어야 합니다.")
    positions = [count // 3, (2 * count) // 3]
    return [int(area.iloc[position]["building_id"]) for position in positions]


def split_held_buildings(data, held_ids):
    """뺀 건물의 전체 기간은 검증, 나머지 건물의 전체 기간은 학습이다."""
    is_held = data["building_id"].isin(held_ids)
    train = data.loc[~is_held].reset_index(drop=True)
    valid = data.loc[is_held].reset_index(drop=True)
    return train, valid


# 전력소비량 0인 행 제외 함수
# 0은 몇 시간씩 연속으로 나타나서 정전·계측 누락으로 판단한다(전체 68행).
# 학습에 남기면 모델이 잘못 배우고, 검증에 남기면 실제로 맞힐 수 없는 값으로 점수가 깎인다.
def remove_zero_power(data):
    is_zero = data[TARGET] == 0
    return data[~is_zero].reset_index(drop=True), int(is_zero.sum())


# 학습·검증 세트 준비 함수 (로드 → 날짜 생성 → 건물 2동 제외 → 0값 제외)
def load_train_valid(building_type, verbose=True):
    data = add_datetime(load_type_data(building_type, "train"))
    held_ids = holdout_building_ids(data)
    train, valid = split_held_buildings(data, held_ids)
    train, train_zero = remove_zero_power(train)
    valid, valid_zero = remove_zero_power(valid)
    if verbose:
        print(
            f"[{building_type}] 학습 {len(train)}행(0값 {train_zero}행 제외), "
            f"검증 {len(valid)}행(0값 {valid_zero}행 제외), 검증 건물 {held_ids}"
        )
    return train, valid


# test.csv 읽기 함수 (정답이 없으므로 최종 예측값 생성용으로만 쓴다)
def load_test(building_type):
    return add_datetime(load_type_data(building_type, "test"))


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

    # Windows에 있는 맑은 고딕만 지정한다. 없는 글꼴을 목록에 넣으면 글자마다 경고가 난다.
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False


if __name__ == "__main__":
    # 공용 모듈 점검: 유형별 검증 건물과 학습·검증이 전체 기간인지 출력한다.
    for building_type in BUILDING_TYPES:
        data = add_datetime(load_type_data(building_type, "train"))
        held_ids = holdout_building_ids(data)
        train, valid = split_held_buildings(data, held_ids)
        print(
            f"{building_type:17s} 검증 {held_ids}  "
            f"학습 {train['building_id'].nunique()}동 {len(train)}행  "
            f"검증 {valid['building_id'].nunique()}동 {len(valid)}행  "
            f"학습기간 {train['date_time'].min()} ~ {train['date_time'].max()}"
        )
