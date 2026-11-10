# 건물 전력소비량 예측

머신러닝 수업 프로젝트. 건물 100동의 시간별 전력소비량(kWh)을 예측한다.

건물 유형마다 결정트리, 베이지안 선형 회귀, 가우시안 프로세스, 인공신경망의 파라미터를 반복해서 찾고, 그 네 모델 중 검증 점수가 가장 좋은 하나를 고른다. 최종 시스템은 유형별로 이렇게 정한 모델을 사용해 테스트 기간의 전력소비량을 예측한다.

## 팀

| 이름 | 역할 | 담당 모델 |
| --- | --- | --- |
| 박준희 | PM | 베이지안 선형 회귀, 인공신경망 |
| 백현승 | Engineer | 결정트리 |
| 우송이 | Analyst | 가우시안 프로세스 |

프로젝트 기간: 2026년 9월 26일 ~ 12월 14일

## 데이터

원본은 `0_raw_data`에 있다.

| 파일 | 내용 |
| --- | --- |
| `train.csv` | 2024-06-01 00시 ~ 2024-08-24 23시. 건물 100동, 85일, 시간 단위 204,000행. 정답 `power_consumption`이 있다 |
| `test.csv` | 2024-08-25 00시 ~ 2024-08-31 23시. 같은 100동, 7일, 16,800행. 정답이 없다 |
| `building_info.csv` | 건물번호, 건물유형, 연면적, 냉방면적, 태양광·ESS·PCS 용량 |

건물 유형은 10개다. 아파트 9동, 상용 10동, 백화점 16동, 병원 9동, 호텔 10동, IDC(전화국) 9동, 건물기타 10동, 공공 8동, 연구소 9동, 학교 10동.

날씨는 기온, 강수량, 풍속, 습도다. 학습 파일에는 일조, 일사도 있으나 테스트 파일에는 없어서 모델 입력에서 뺀다. 요일은 월요일 0, 일요일 6이다.

`2_data_codes/0_data_processing.py`가 건물 정보를 붙이고 유형별 CSV로 나눈 결과가 `1_source_data/train`, `1_source_data/test`다. 유형 파일 이름은 `apartment`, `commercial`, `department_store`, `hospital`, `hotel`, `idc`, `other`, `public`, `research`, `school`이다.

### 학습과 검증

테스트 파일에는 정답이 없으므로, 모델 비교는 학습 기간 85일을 날짜로 나눠 한다.

`2024-08-08 00시` 미만은 학습(6월 1일 ~ 8월 7일, 68일, 80%)이고, 그 시각부터는 검증(8월 8일 ~ 8월 24일, 17일, 20%)이다. 행 순서로 80%를 자르면 기간이 아니라 건물이 잘리므로 날짜로 자른다. 이 기준은 `4_source_codes/0_common.py`의 `SPLIT_DATE`에 있다.

파라미터 탐색과 모델 선정은 이 검증 점수로만 한다. 점수가 정해진 뒤 85일 전체로 다시 학습하지 않는다. 비교 지표는 MAE, RMSE, R²다.

전력소비량이 0인 행은 정전이나 계측 누락으로 보고, 공용 로더에서 학습·검증 모두 제외한다.

## 디렉토리

```text
ML_test
├── 0_raw_data                  원본 CSV
│   ├── train.csv
│   ├── test.csv
│   └── building_info.csv
├── 1_source_data               유형별로 나눈 학습·테스트 CSV
│   ├── train
│   └── test
├── 2_data_codes                전처리와 데이터 확인
│   ├── 0_data_processing.py    원본을 유형별 CSV로 저장
│   ├── 1_data_cleaning.py      결측, 중복, 이상치 확인. 값은 바꾸지 않음
│   └── 2_data_analyzing.py     유형·시간·면적·날씨와 사용량의 관계 확인
├── 3_models_train_codes        모델 학습 함수. 분할, 점수, 저장은 하지 않음
│   ├── 0_bayesian_regression.py
│   ├── 1_decision_tree_regression.py
│   ├── 2_gaussian.py
│   └── 3_artificial_neural_network.py
├── 4_source_codes              공통 처리와 실행 스크립트
│   ├── 0_common.py             로드, 날짜 분할, 0값 제외, MAE·RMSE·R²
│   ├── 1_total_script.py       유형별 파라미터 탐색, 모델 선정, 저장
│   └── 2_final_system.py       유형별로 선정된 모델로 예측하는 최종 실행
├── 5_models                    학습이 끝난 모델 파일
├── reference                   담당자가 먼저 만든 원본 코드와 결과
└── README.md
```

학습 함수는 넘겨받은 `X`, `y`, 파라미터로 한 번만 학습한다. 유형별 분할, 파라미터 반복, 점수 계산, 모델 저장은 `1_total_script.py`에서 맡는다. 인공신경망 학습 코드, 통합 스크립트, 최종 실행 파일은 아직 비어 있다.

`reference/Bayesian`은 박준희의 베이지안 원본과 회의 메모 `DECISIONS.md`다. `reference/가우시안_송이`는 우송이의 가우시안 원본, 실험 기록, 검증 결과다. `reference/tree_test.py`는 저장된 결정트리로 한 시각을 예측하는 예시이다. 지금 파이프라인이 실행하는 코드는 `3_models_train_codes`와 `4_source_codes`다.

전처리 스크립트와 `0_common.py`의 경로 문자열은 `raw_data`, `source_data`다. 현재 폴더 이름은 `0_raw_data`, `1_source_data`이므로, 스크립트를 실행하기 전에 이 경로를 맞춘다.

## 모델 선정

한 건물 유형에 대해 아래 순서로 정하고, 같은 과정을 10개 유형에 반복한다.

1. 모델 4개 각각에 대해 파라미터를 반복문으로 바꿔 가며 학습한다. 검증 점수가 가장 좋은 파라미터를 그 모델의 설정으로 둔다.
2. 유형 하나에 최적 파라미터가 적용된 모델 4개가 생긴다. 그중 검증 점수가 가장 좋은 모델을 그 유형의 최종 모델로 고른다.
3. 최종 시스템은 그 건물 유형에 이 모델을 사용한다.

10개 유형이 끝나면 전체 시스템 구성이 끝난다. `2_final_system.py`는 유형마다 여기서 고른 모델을 불러 예측한다.

| 모델 | 파일 | 반복해서 찾는 값 |
| --- | --- | --- |
| 결정트리 | `1_decision_tree_regression.py` | `max_depth`, `min_samples_leaf` 등 |
| 베이지안 선형 회귀 | `0_bayesian_regression.py` | `alpha`, `beta` 등 |
| 가우시안 프로세스 | `2_gaussian.py` | `trend`, `restarts` 등 |
| 인공신경망 | `3_artificial_neural_network.py` | 작성 전 |

학습 함수는 넘겨받은 파라미터로 한 번 학습해 모델을 반환한다. 파라미터를 바꿔 호출하는 반복은 `1_total_script.py`가 맡는다. 가우시안 프로세스는 계산량이 행 수의 세제곱이라, 한 유형 안에서는 건물 1동씩 학습한다.

파일 이름이 숫자로 시작하므로 학습 함수는 이렇게 불러온다.

```python
import importlib

tree = importlib.import_module("1_decision_tree_regression")
model = tree.train_tree(X, y)
```

## 실행 환경

Python에서 pandas, numpy, scikit-learn을 쓴다. 데이터 분석 그래프에는 matplotlib, seaborn이 필요하다.
