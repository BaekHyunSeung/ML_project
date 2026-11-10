from sklearn.tree import DecisionTreeRegressor


def train_tree(X, y, max_depth):
    """전달받은 입력, 정답, 깊이로 결정트리를 학습해 반환한다.

    데이터 분할, 점수 계산, 파일 저장은 하지 않는다.
    """
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X, y)
    return model

# 호출 예시
# from tree import train_tree
# model = train_tree(X, y, max_depth=10)