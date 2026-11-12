# 통합 스크립트
# 건물 유형 10개를 순서대로 처리한다.
# 유형마다 데이터를 나누고, 파라미터를 찾고, 모델을 저장한 뒤 가장 좋은 모델을 고른다.
# 선정 결과는 종료 전에 5_models/model_select.txt 에 남긴다.

import importlib

common = importlib.import_module("0_common")
parameter_search = importlib.import_module("2_parameter_search")
model_train = importlib.import_module("1_model_train")


def split_one_type(building_type):
    """건물 유형 하나에서 연면적 기준 2동을 검증으로 뺀다.

    학습은 나머지 건물의 전체 기간이고, 점수는 뺀 2동의 전체 기간으로 계산한다.
    """
    data = common.add_datetime(common.load_type_data(building_type, "train"))
    held_ids = common.holdout_building_ids(data)
    train, valid = common.split_held_buildings(data, held_ids)
    print(f"검증 건물: {held_ids}")
    return train, valid


def save_selected_models(building_type, train, best, rows_per_building=None):
    """탐색에서 고른 파라미터로 세 모델을 학습해 저장한다."""
    tree_path = model_train.save_tree(
        train, best[model_train.tree.MODEL_NAME], building_type
    )
    bayesian_path = model_train.save_bayesian(
        train, best[model_train.BAYESIAN_NAME], building_type
    )
    gaussian_kwargs = {}
    if rows_per_building is not None:
        gaussian_kwargs["rows_per_building"] = rows_per_building
    gaussian_path = model_train.save_gaussian(
        train, best[model_train.GAUSSIAN_NAME], building_type, **gaussian_kwargs
    )
    return {
        model_train.tree.MODEL_NAME: tree_path,
        model_train.BAYESIAN_NAME: bayesian_path,
        model_train.GAUSSIAN_NAME: gaussian_path,
    }


def run_one_type(building_type, **search_kwargs):
    """한 건물 유형의 파라미터를 찾고, 모델을 저장한 뒤 가장 좋은 모델을 고른다."""
    train, valid = split_one_type(building_type)
    best, table = parameter_search.find_parameters(
        train, valid, building_type=building_type, **search_kwargs
    )
    paths = save_selected_models(
        building_type,
        train,
        best,
        rows_per_building=search_kwargs.get("rows_per_building"),
    )
    selected_model, selected_params, compared, selection_image = parameter_search.select_model(
        table, building_type
    )
    return best, table, paths, selected_model, selected_params, compared, selection_image


def model_select_path():
    """유형별 선정 모델과 파라미터를 적을 파일 경로."""
    return model_train.SAVED_MODELS_DIR / "model_select.txt"


def save_model_selection(records):
    """건물 유형별 선정 모델과 파라미터를 5_models/model_select.txt에 저장한다."""
    path = model_select_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["건물유형\t한글이름\t모델\t파라미터"]
    for building_type, model_name, params in records:
        type_name = common.BUILDING_TYPE_NAMES.get(building_type, building_type)
        param_text = ", ".join(f"{key}={value}" for key, value in params.items())
        lines.append(f"{building_type}\t{type_name}\t{model_name}\t{param_text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"저장: {path}")
    return path


def run_all_types(**search_kwargs):
    """건물 유형 10개를 순서대로 학습하고, 유형별 선정 모델을 반환한다.

    각 유형이 끝날 때마다 model_select.txt를 갱신하고, 전체가 끝나면 그 파일이 최종본이다.
    """
    selected = {}
    records = []
    for building_type in common.BUILDING_TYPES:
        type_name = common.BUILDING_TYPE_NAMES.get(building_type, building_type)
        print(f"시작: {type_name}")
        _, _, _, model_name, params, _, _ = run_one_type(building_type, **search_kwargs)
        selected[building_type] = model_name
        records.append((building_type, model_name, params))
        save_model_selection(records)
        label = parameter_search.MODEL_NAMES.get(model_name, model_name)
        print(f"완료: {type_name} -> {label}")
    return selected


if __name__ == "__main__":
    run_all_types()
