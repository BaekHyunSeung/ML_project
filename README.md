# 건물 전력소비량 예측

머신러닝 수업 프로젝트. 건물 100동의 시간별 전력소비량(kWh)을 예측한다.

건물 유형마다 결정트리, 베이지안 선형 회귀, 가우시안 프로세스의 파라미터를 찾고, 검증 점수가 가장 좋은 하나를 그 유형의 사용 모델로 고른다. 인공신경망 학습 코드와 최종 예측 실행 파일은 아직 비어 있다.

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

테스트 파일에는 정답이 없으므로, 모델 비교는 유형 안에서 건물 2동을 빼서 한다.

연면적이 작은 순으로 1/3, 2/3 위치의 2동을 검증으로 뺀다. 학습은 나머지 건물의 85일 전체이고, 점수는 뺀 2동의 85일 전체로 계산한다. 가장 작은 건물과 가장 큰 건물은 학습에 남긴다. 검증에 쓴 2동은 모델을 저장할 때 다시 넣지 않는다. 비교 지표는 MAE, RMSE, R²다. MAE가 가장 작은 쪽을 고르고, MAE가 같으면 RMSE가 더 작은 쪽을 고른다. 유형끼리 MAE는 비교하지 않는다.

| 유형 | 제외 건물 번호 |
| --- | --- |
| 아파트 | 11, 25 |
| 상용 | 51, 16 |
| 백화점 | 32, 73 |
| 병원 | 90, 75 |
| 호텔 | 1, 9 |
| IDC(전화국) | 64, 52 |
| 건물기타 | 82, 78 |
| 공공 | 72, 33 |
| 연구소 | 49, 15 |
| 학교 | 22, 5 |

전력소비량이 0인 행은 정전이나 계측 누락으로 본다. `load_train_valid`는 이 행을 학습·검증에서 빼지만, `3_total_script.py`의 분할은 건물만 나누고 0값 행은 그대로 둔다.

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
│   ├── 0_common.py             로드, 건물 2동 제외, 0값 제외, MAE·RMSE·R²
│   ├── 1_model_train.py        모델별 점수 계산과 모델 저장
│   ├── 2_parameter_search.py   후보 비교, 점수 이미지, 사용 모델 선정
│   ├── 3_total_script.py       10개 유형 분할, 탐색, 저장, 선정
│   └── 4_final_system.py       유형별 사용 모델로 예측. 아직 비어 있음
├── 5_models                    학습이 끝난 모델과 선정 결과
│   └── model_select.txt
├── results                     파라미터 표와 모델 선정 표 이미지
│   ├── parameter_search
│   └── model_selection
├── reference                   담당자가 먼저 만든 원본 코드와 결과
└── README.md
```

학습 함수는 넘겨받은 `X`, `y`, 파라미터로 한 번만 학습한다. `3_total_script.py`가 유형을 나누고 `2_parameter_search.py`로 파라미터를 찾은 뒤 `1_model_train.py`에 저장을 맡긴다. 이어서 세 모델 중 MAE가 가장 작은 모델을 고르고 `5_models/model_select.txt`에 적는다.

`reference/Bayesian`은 박준희의 베이지안 원본과 회의 메모 `DECISIONS.md`다. `reference/가우시안_송이`는 우송이의 가우시안 원본, 실험 기록, 검증 결과다. `reference/tree_test.py`는 저장된 결정트리로 한 시각을 예측하는 예시이다. 지금 파이프라인이 실행하는 코드는 `3_models_train_codes`와 `4_source_codes`다.

`0_common.py`는 `1_source_data`를 읽는다. 전처리 스크립트의 경로 문자열은 아직 `raw_data`, `source_data`다.

## 모델 선정

한 건물 유형에 대해 아래 순서로 정하고, 같은 과정을 10개 유형에 반복한다.

1. 결정트리, 베이지안, 가우시안 각각에 대해 정해 둔 후보를 학습하고, 뺀 건물 2동의 MAE가 가장 작은 파라미터를 그 모델의 설정으로 둔다.
2. 그 파라미터로 남은 건물의 85일 전체를 다시 학습해 모델 파일 세 개를 저장한다. 가우시안은 이때도 건물당 150행만 뽑는다.
3. 세 모델의 검증 MAE를 비교해 그 유형의 사용 모델을 고른다. 후보 전체 표와 선정 표는 이미지로 저장한다.

| 모델 | 파일 | 후보 |
| --- | --- | --- |
| 결정트리 | `1_decision_tree_regression.py` | `max_depth` 5, 10, 15 × `min_samples_leaf` 1, 5. `random_state`는 42 |
| 베이지안 선형 회귀 | `0_bayesian_regression.py` | `alpha` 0.01, 0.0001, 0.000001 × `beta` 0.01, 0.0001 |
| 가우시안 프로세스 | `2_gaussian.py` | `trend` none, smooth, linear. `restarts`는 0. 건물당 150행 |
| 인공신경망 | `3_artificial_neural_network.py` | 작성 전. 선정 반복에 들어가지 않음 |

가우시안 프로세스는 건물 유형 하나당 모델 하나다. 학습은 `log1p`로 하고, 예측은 `expm1`로 kWh에 되돌린다.

파일 이름이 숫자로 시작하므로 학습 함수는 이렇게 불러온다.

```python
import importlib

tree = importlib.import_module("1_decision_tree_regression")
model = tree.train_tree(X, y)
```

### 저장 위치

공공 유형을 예로 든다. 다른 유형은 파일 이름의 `public`만 바뀐다.

| 내용 | 경로 |
| --- | --- |
| 파라미터 검증 표 | `results/parameter_search/public.png` |
| 모델 선정 표 | `results/model_selection/public.png` |
| 결정트리 | `5_models/decision_tree/public.joblib` |
| 베이지안 | `5_models/bayesian_linear_regression/public.joblib` |
| 가우시안 | `5_models/gaussian_process/public.joblib` |
| 유형별 사용 모델 | `5_models/model_select.txt` |

세 모델 파일을 모두 저장한 뒤 하나를 고른다. 고르지 않은 파일도 지우지 않는다. `model_select.txt`의 열은 건물유형, 한글이름, 모델, 파라미터다.

현재 선정 결과는 다음과 같다.

| 유형 | 사용 모델 | 파라미터 |
| --- | --- | --- |
| 아파트 | 베이지안 | `alpha=0.01`, `beta=0.0001` |
| 상용 | 결정트리 | `max_depth=5`, `min_samples_leaf=1` |
| 백화점 | 결정트리 | `max_depth=15`, `min_samples_leaf=5` |
| 병원 | 베이지안 | `alpha=0.01`, `beta=0.0001` |
| 호텔 | 가우시안 | `trend=smooth`, `restarts=0` |
| IDC(전화국) | 가우시안 | `trend=smooth`, `restarts=0` |
| 건물기타 | 베이지안 | `alpha=1e-06`, `beta=0.01` |
| 공공 | 가우시안 | `trend=none`, `restarts=0` |
| 연구소 | 베이지안 | `alpha=0.01`, `beta=0.0001` |
| 학교 | 가우시안 | `trend=linear`, `restarts=0` |

## 실행

`4_source_codes`에서 실행한다. 10개 유형이 순서대로 학습되고, 이미지와 모델 파일, `model_select.txt`가 다시 저장된다.

```powershell
python 3_total_script.py
```

프로젝트 폴더에서 실행할 때는 경로를 적는다.

```powershell
python "4_source_codes/3_total_script.py"
```

가우시안이 유형마다 세 번 학습하므로 전부 끝나는 데는 시간이 걸린다.

## 실행 환경

Python에서 pandas, numpy, scikit-learn, joblib을 쓴다. 점수 표 이미지에는 matplotlib이 필요하고, 한글은 Malgun Gothic을 쓴다. 데이터 분석 그래프에는 seaborn도 쓴다.
