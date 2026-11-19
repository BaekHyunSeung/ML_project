인공신경망 (박준희) - 팀 공유 폴더 (2026-11-17)

[이 폴더만으로 실행됩니다]
source_data(유형별 전처리 CSV)가 들어 있어서 폴더를 어디에 두든 바로 실행됩니다.
폴더 안의 구조(source_code_file, source_data, results)만 바꾸지 말아주세요.

[실행]
  cd source_code_file
  python 3_neural_network.py public   -> 공공 유형 하나만 학습 + 정상 작동 점검 (약 10초)
  python 3_neural_network.py          -> 10개 유형 전체 학습 + 결과 CSV + 그래프 저장 (약 30초)
필요 라이브러리: numpy, pandas, scikit-learn, matplotlib
random_state를 고정해서 다시 실행해도 같은 결과가 나옵니다.

[common.py가 바뀌었습니다 - 꼭 교체해 주세요]
각자 Project/source_code_file/common.py를 이 폴더의 common.py로 바꿔주세요.
- 추가된 함수(팀 입력 명세): make_base_features, drop_constant_columns, standardize,
  add_time_dummies, add_weekend, pick_representative_building, fix_humidity_zero
- 기존 함수(load_train_valid, regression_metrics, save_results 등)는 이름과 동작이 그대로라
  지금까지 짠 코드는 그대로 돌아갑니다. 이전 베이지안 코드도 새 common.py로 그대로 동작합니다.
- 기본 비교용 입력은 make_base_features로 만들면 네 모델이 같은 입력을 쓰게 됩니다.
    from common import load_train_valid, make_base_features, drop_constant_columns, standardize
    train, valid = load_train_valid("hotel")
    train_x, valid_x = make_base_features(train), make_base_features(valid)
    train_x, valid_x, dropped = drop_constant_columns(train_x, valid_x)
    train_x, valid_x = standardize(train_x, valid_x)   # 트리는 표준화 안 해도 됨

[파일 설명]
- DECISIONS.md : 결정 사항. C절(19~25번)이 신경망 관련 추가 내용, 'O' 표시는 회의 안건
- source_code_file/common.py : 팀 공용 모듈 (위 설명 참고)
- source_code_file/3_neural_network.py : 인공신경망 학습 코드 (sklearn MLPRegressor)
- results/neural_network_results.csv : 유형별 성능
    공통 열 model, building_type, n_train, n_valid, MAE, RMSE, R2 + 신경망 전용 열 n_epochs, converged, hidden_layers
- results/neural_network_summary.md : 결과 요약
- results/neural_network_*.png : 유형별 예측 그래프 10개, R2 막대, 베이지안 비교, 학습 곡선
- results/bayesian_results.csv : 베이지안 결과 (신경망 실행 시 비교 그래프를 그리는 데 사용)

[참고_개선실험 폴더]
팀 입력 명세 밖의 실험(면적·설비 열 제거, 건물번호, 날씨·주기형 시간·공휴일 추가)입니다. 정식 비교 결과가 아닙니다.
실행해 보려면 안의 파일을 같은 위치(source_code_file, results)로 옮기면 됩니다. (약 2~3분)
