# 최종 예측
# 실행하면 선정된 모델을 불러 두고, 사용자가 건물 유형과 값을 넣을 때까지 기다린다.
# 건물 유형을 고르면 그 유형의 모델과 필요한 입력을 보여 주고, 그 시각의 전력소비량을 예측한다.

import importlib
import sys
from pathlib import Path

import pandas as pd

SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

common = importlib.import_module("0_common")
model_train = importlib.import_module("1_model_train")
parameter_search = importlib.import_module("2_parameter_search")

BUILDING_FIELDS = [
    ("total_area", "연면적(m²)"),
    ("cooling_area", "냉방면적(m²)"),
    ("solar_capacity", "태양광 용량"),
    ("ess_capacity", "ESS 용량"),
    ("pcs_capacity", "PCS 용량"),
]
WEATHER_FIELDS = {
    "temperature": "기온(°C)",
    "precipitation": "강수량",
    "wind_speed": "풍속",
    "humidity": "습도(%)",
}
TIME_FIELDS = [
    ("year", "연도"),
    ("month", "월"),
    ("day", "일"),
    ("hour", "시"),
]
MODEL_INPUTS = {
    "decision_tree": [
        "total_area",
        "cooling_area",
        "solar_capacity",
        "ess_capacity",
        "pcs_capacity",
        "temperature",
        "humidity",
        "wind_speed",
    ],
    "bayesian_linear_regression": [
        "total_area",
        "cooling_area",
        "solar_capacity",
        "ess_capacity",
        "pcs_capacity",
        "temperature",
    ],
    "neural_network": [
        "total_area",
        "cooling_area",
        "solar_capacity",
        "ess_capacity",
        "pcs_capacity",
        "temperature",
    ],
    "gaussian_process": [
        "total_area",
        "cooling_area",
        "solar_capacity",
        "ess_capacity",
        "pcs_capacity",
        "temperature",
        "precipitation",
        "wind_speed",
        "humidity",
    ],
}
FIELD_LABELS = dict(BUILDING_FIELDS)
FIELD_LABELS.update(WEATHER_FIELDS)


def load_selection():
    """model_select.txt의 유형별 선정 모델과 저장 파일을 읽는다."""
    path = model_train.model_file("decision_tree", "public").parent.parent / "model_select.txt"
    if not path.exists():
        raise FileNotFoundError(f"선정 결과가 없습니다: {path}")
    table = pd.read_csv(path, sep="\t")
    selected = {}
    for _, row in table.iterrows():
        building_type = row["건물유형"]
        model_name = row["모델"]
        model_path = model_train.model_file(model_name, building_type)
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")
        selected[building_type] = {
            "name": row["한글이름"],
            "model_name": model_name,
            "label": parameter_search.MODEL_NAMES.get(model_name, model_name),
            "params": row["파라미터"],
            "model": model_train.joblib.load(model_path),
        }
    return selected


def input_fields(model_name):
    """그 모델이 사용자에게 받을 값. 요일과 시간대 변환은 날짜로 만든다."""
    return MODEL_INPUTS[model_name] + [key for key, _ in TIME_FIELDS]


def describe_inputs(model_name):
    labels = [FIELD_LABELS[key] for key in MODEL_INPUTS[model_name]]
    return "건물·날씨: " + ", ".join(labels) + " / 시간: 연도, 월, 일, 시"


def ask_number(label, minimum=None, maximum=None):
    while True:
        text = input(f"{label}: ").strip()
        if text.lower() == "q":
            return None
        try:
            value = float(text)
        except ValueError:
            print("숫자를 입력하세요. 종료는 q입니다.")
            continue
        if minimum is not None and value < minimum:
            print(f"{minimum} 이상이어야 합니다.")
            continue
        if maximum is not None and value > maximum:
            print(f"{maximum} 이하여야 합니다.")
            continue
        return value


def ask_values(model_name):
    """모델에 필요한 값을 하나씩 받는다. q이면 종료한다."""
    limits = {"month": (1, 12), "day": (1, 31), "hour": (0, 23), "year": (2024, 2100)}
    values = {}
    for key in input_fields(model_name):
        label = FIELD_LABELS.get(key) or dict(TIME_FIELDS)[key]
        minimum, maximum = limits.get(key, (None, None))
        value = ask_number(label, minimum, maximum)
        if value is None:
            return None
        values[key] = value
    when = pd.Timestamp(
        year=int(values["year"]),
        month=int(values["month"]),
        day=int(values["day"]),
        hour=int(values["hour"]),
    )
    values["when"] = when
    return values


def prediction_row(model_name, values):
    """입력값을 그 모델의 predict가 읽는 한 행으로 만든다."""
    when = values["when"]
    row = {
        "total_area": values.get("total_area", 0.0),
        "cooling_area": values.get("cooling_area", 0.0),
        "solar_capacity": values.get("solar_capacity", 0.0),
        "ess_capacity": values.get("ess_capacity", 0.0),
        "pcs_capacity": values.get("pcs_capacity", 0.0),
        "temperature": values.get("temperature", 0.0),
        "precipitation": values.get("precipitation", 0.0),
        "wind_speed": values.get("wind_speed", 0.0),
        "humidity": values.get("humidity", 0.0),
        "month": when.month,
        "hour": when.hour,
        "weekday": int(when.dayofweek),
        "num_date_time": f"0_{when.strftime('%Y%m%d %H')}",
    }
    return pd.DataFrame([row])


def predict_power(model_name, model, values):
    row = prediction_row(model_name, values)
    if model_name == "decision_tree":
        columns = list(model.feature_names_in_)
        predicted = model.predict(row[columns])
    else:
        predicted = model.predict(row)
    return float(predicted[0])


def choose_type(selected):
    print("\n건물 유형을 고르세요. 종료는 q입니다.")
    options = []
    for index, building_type in enumerate(common.BUILDING_TYPES, start=1):
        if building_type not in selected:
            continue
        item = selected[building_type]
        options.append(building_type)
        print(f"{index}. {item['name']}")
    while True:
        text = input("번호: ").strip()
        if text.lower() == "q" or text == "":
            return None
        if text.isdigit() and 1 <= int(text) <= len(options):
            return options[int(text) - 1]
        print("목록에 있는 번호를 입력하세요.")


def run():
    selected = load_selection()
    print("선정된 모델을 불러왔습니다. 건물 유형을 고르면 그 시각의 전력소비량을 예측합니다.")
    while True:
        building_type = choose_type(selected)
        if building_type is None:
            print("종료합니다.")
            return
        item = selected[building_type]
        print(f"\n{item['name']}은 {item['label']}을 사용합니다. ({item['params']})")
        print("필요한 정보:", describe_inputs(item["model_name"]))
        print("요일은 연·월·일로 계산합니다. 종료는 q입니다.")
        values = ask_values(item["model_name"])
        if values is None:
            print("종료합니다.")
            return
        predicted = predict_power(item["model_name"], item["model"], values)
        when = values["when"].strftime("%Y-%m-%d %H시")
        print(f"{item['name']} {when} 예측 전력소비량: {predicted:.1f} kWh")


if __name__ == "__main__":
    run()
