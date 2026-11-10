#임시로 결정 트리 사용 코드 만듬
from pathlib import Path

import joblib
import pandas as pd

# 건물 유형을 고르면 그 유형에서 쓸 모델 파일을 찾는다.
# 지금은 IDC 결정트리만 저장되어 있다.
MODEL_FILES = {
    "IDC(전화국)": Path("models/idc_decision_tree.joblib"),
}


def predict_power(
    building_type,
    total_area,
    cooling_area,
    solar_capacity,
    ess_capacity,
    pcs_capacity,
    temperature,
    humidity,
    wind_speed,
    when,
):
    """건물 유형, 건물 정보, 날씨, 시각 하나로 그 시각의 전력소비량을 예측한다."""
    model_path = MODEL_FILES.get(building_type)
    if model_path is None or not model_path.exists():
        known = ", ".join(MODEL_FILES)
        raise FileNotFoundError(f"{building_type} 모델이 없습니다. 저장된 유형: {known}")

    saved = joblib.load(model_path)
    when = pd.Timestamp(when)
    row = pd.DataFrame([{
        "total_area": total_area,
        "cooling_area": cooling_area,
        "solar_capacity": solar_capacity,
        "ess_capacity": ess_capacity,
        "pcs_capacity": pcs_capacity,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "month": when.month,
        "hour": when.hour,
        "weekday": when.dayofweek,
    }])
    return float(saved["model"].predict(row[saved["feature_cols"]])[0])


if __name__ == "__main__":
    # 건물 30, 2024-08-25 00시 한 건만 예측한다.
    predicted = predict_power(
        building_type="IDC(전화국)",
        total_area=31156.75,
        cooling_area=13487.0,
        solar_capacity=0.0,
        ess_capacity=0.0,
        pcs_capacity=0.0,
        temperature=25.0,
        humidity=90.0,
        wind_speed=0.2,
        when="2024-08-25 00:00",
    )
    print("건물유형: IDC(전화국)")
    print("시각: 2024-08-25 00:00")
    print(f"예측 전력소비량: {predicted:.3f}")
