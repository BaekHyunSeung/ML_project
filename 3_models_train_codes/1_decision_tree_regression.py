# 결정트리 회귀 학습
# 전달받은 입력과 정답으로 학습만 한다.
# 데이터 분할, 점수 계산, 파일 저장은 하지 않는다.
# 열을 고르거나 스케일을 맞추지 않는다. 호출하는 쪽이 이미 맞춘 X, y를 넘긴다.
#
# 호출 예시
#   import importlib
#   tree = importlib.import_module("1_decision_tree_regression")
#   model = tree.train_tree(X, y)

from sklearn.tree import DecisionTreeRegressor

MODEL_NAME = "decision_tree"

# 모든 건물 유형에 같은 값을 쓴다.
MAX_DEPTH = 10
MIN_SAMPLES_LEAF = 1
CRITERION = "squared_error"
RANDOM_STATE = 42


def train_tree(
    X,
    y,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    criterion=CRITERION,
    random_state=RANDOM_STATE,
):
    """전달받은 입력과 정답으로 결정트리를 학습해 반환한다.

    데이터 분할, 점수 계산, 파일 저장은 하지 않는다.
    인자를 생략하면 깊이 10, 잎의 최소 행 수 1, 제곱오차, random_state 42로 학습한다.
    """
    model = DecisionTreeRegressor(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X, y)
    return model
