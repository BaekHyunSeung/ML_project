# 파라미터 탐색
# split_one_type으로 나눈 학습·검증 데이터로 후보 파라미터를 비교한다.
# 표의 한 행은 입력 파라미터와 검증 MAE, RMSE, R²다. 모델 파일은 만들지 않는다.
# 같은 표는 results/parameter_search/{건물유형}.png 로도 저장한다.

import importlib

import pandas as pd

common = importlib.import_module("0_common")
model_train = importlib.import_module("1_model_train")

TREE_CANDIDATES = [
    {"max_depth": depth, "min_samples_leaf": leaf}
    for depth in (5, 10, 15)
    for leaf in (1, 5)
]
BAYESIAN_CANDIDATES = [
    {"alpha": alpha, "beta": beta}
    for alpha in (1e-2, 1e-4, 1e-6)
    for beta in (1e-2, 1e-4)
]
GAUSSIAN_CANDIDATES = [
    {"trend": trend, "restarts": 0}
    for trend in ("none", "smooth", "linear")
]
METRIC_COLUMNS = ("MAE", "RMSE", "R2")
MODEL_NAMES = {
    "decision_tree": "결정트리",
    "bayesian_linear_regression": "베이지안",
    "gaussian_process": "가우시안",
}
IMAGE_COLUMNS = ("모델 이름", "입력 파라미터", "MAE", "RMSE", "R²")


def _order_columns(table):
    metrics = [column for column in METRIC_COLUMNS if column in table.columns]
    others = [column for column in table.columns if column not in metrics and column != "model"]
    return table[["model"] + others + metrics]


def _format_value(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _parameter_text(row):
    parts = []
    for key, value in row.items():
        if key in ("model", *METRIC_COLUMNS) or pd.isna(value):
            continue
        parts.append(f"{key}={_format_value(value)}")
    return ", ".join(parts)


def parameter_image_table(table):
    """이미지에 넣을 열만 남긴다. 모델 이름, 입력 파라미터, MAE, RMSE, R²."""
    rows = []
    for _, row in table.iterrows():
        rows.append(
            {
                "모델 이름": MODEL_NAMES.get(row["model"], row["model"]),
                "입력 파라미터": _parameter_text(row),
                "MAE": f"{row['MAE']:.4f}",
                "RMSE": f"{row['RMSE']:.4f}",
                "R²": f"{row['R2']:.4f}",
            }
        )
    return pd.DataFrame(rows, columns=IMAGE_COLUMNS)


def _draw_score_table(shown, path, title=None, selected_name=None):
    """모델 이름, 입력 파라미터, MAE, RMSE, R² 표를 PNG로 저장한다."""
    common.setup_korean_font()
    import matplotlib.pyplot as plt

    row_count = len(shown)
    fig, ax = plt.subplots(figsize=(14, max(2.4, 0.48 * row_count + 1.1)))
    ax.axis("off")
    if title:
        ax.set_title(title, pad=12)
    drawn = ax.table(
        cellText=shown.to_numpy(),
        colLabels=list(shown.columns),
        loc="center",
        cellLoc="center",
        colWidths=[0.14, 0.48, 0.12, 0.13, 0.13],
    )
    drawn.auto_set_font_size(False)
    drawn.set_fontsize(10)
    drawn.scale(1, 1.5)
    for (row_index, _), cell in drawn.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", weight="bold")
        elif selected_name is not None and shown.iloc[row_index - 1]["모델 이름"] == selected_name:
            cell.set_facecolor("#c6efce")
        elif row_index % 2 == 0:
            cell.set_facecolor("#f3f6f9")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {path}")
    return path


def save_parameter_image(table, building_type=None):
    """파라미터 점수 표를 PNG로 저장한다."""
    shown = parameter_image_table(table)
    folder = common.RESULTS_DIR / "parameter_search"
    filename = f"{building_type}.png" if building_type else "parameter_search.png"
    title = None
    if building_type:
        type_name = common.BUILDING_TYPE_NAMES.get(building_type, building_type)
        title = f"{type_name} 파라미터 검증 결과"
    return _draw_score_table(shown, folder / filename, title)


def find_parameters(
    train,
    valid,
    tree_candidates=None,
    bayesian_candidates=None,
    gaussian_candidates=None,
    rows_per_building=None,
    building_type=None,
):
    """나뉜 학습·검증 데이터로 모델별 파라미터를 찾고 점수 표를 반환한다."""
    tree_candidates = TREE_CANDIDATES if tree_candidates is None else tree_candidates
    bayesian_candidates = BAYESIAN_CANDIDATES if bayesian_candidates is None else bayesian_candidates
    gaussian_candidates = GAUSSIAN_CANDIDATES if gaussian_candidates is None else gaussian_candidates
    gaussian_kwargs = {}
    if rows_per_building is not None:
        gaussian_kwargs["rows_per_building"] = rows_per_building

    tree_best, tree_table = model_train.search_tree(train, valid, tree_candidates)
    bayesian_best, bayesian_table = model_train.search_bayesian(train, valid, bayesian_candidates)
    gaussian_best, gaussian_table = model_train.search_gaussian(
        train, valid, gaussian_candidates, **gaussian_kwargs
    )

    table = pd.concat(
        [
            tree_table.assign(model=model_train.tree.MODEL_NAME),
            bayesian_table.assign(model=model_train.BAYESIAN_NAME),
            gaussian_table.assign(model=model_train.GAUSSIAN_NAME),
        ],
        ignore_index=True,
    )
    table = _order_columns(table)
    print(table.to_string(index=False))
    save_parameter_image(table, building_type)
    best = {
        model_train.tree.MODEL_NAME: tree_best,
        model_train.BAYESIAN_NAME: bayesian_best,
        model_train.GAUSSIAN_NAME: gaussian_best,
    }
    return best, table


def saved_model_rows(table):
    """모델마다 저장된 파라미터, 즉 검증 MAE가 가장 작은 행만 남긴다."""
    chosen = []
    for _, group in table.groupby("model", sort=False):
        ordered = group.sort_values(["MAE", "RMSE"], ascending=True)
        chosen.append(ordered.iloc[0])
    return pd.DataFrame(chosen).reset_index(drop=True)


def select_model(table, building_type=None):
    """저장된 모델 중 검증 점수가 가장 좋은 모델을 고르고 비교 이미지를 저장한다.

    MAE가 작은 쪽을 고르고, MAE가 같으면 RMSE가 작은 쪽을 고른다.
    """
    compared = saved_model_rows(table)
    best_index = compared.sort_values(["MAE", "RMSE"], ascending=True).index[0]
    chosen = compared.loc[best_index]
    model_name = chosen["model"]
    params = {}
    for key, value in chosen.items():
        if key in ("model", *METRIC_COLUMNS) or pd.isna(value):
            continue
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        params[key] = value
    shown = parameter_image_table(compared)
    selected_label = MODEL_NAMES.get(model_name, model_name)
    folder = common.RESULTS_DIR / "model_selection"
    filename = f"{building_type}.png" if building_type else "model_selection.png"
    type_name = common.BUILDING_TYPE_NAMES.get(building_type, building_type) if building_type else ""
    title = f"{type_name} 선정 모델: {selected_label}".strip()
    save_path = _draw_score_table(shown, folder / filename, title, selected_label)
    print(f"선정: {selected_label} {params}")
    return model_name, params, compared, save_path
