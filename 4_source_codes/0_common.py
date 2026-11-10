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
