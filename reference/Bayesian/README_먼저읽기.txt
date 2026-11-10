베이지안 선형 회귀 (박준희) - 팀 공유 파일

[압축 푸는 위치]
아무 폴더에나 풀어도 됩니다. source_data(유형별 전처리 CSV)가 들어 있어서 이 압축 파일만으로 실행됩니다.
압축 안의 폴더 구조(source_code_file, source_data, results)만 그대로 유지해 주세요.
  예) 베이지안_박준희_팀공유/source_code_file/3_bayesian_regression.py
      베이지안_박준희_팀공유/source_data/train/*.csv, test/*.csv
기존 Project 폴더에 풀 경우 source_data는 팀원 파일과 같은 내용이라 덮어써도 됩니다.

[파일 설명]
- DECISIONS.md : 회의 안건. 'O' 표시는 팀 합의가 필요한 항목
- source_code_file/common.py : 팀 공용 모듈 (데이터 로드, 날짜 기준 분할, 0값 제외, MAE/RMSE/R2, 결과 저장)
    사용 예) from common import load_train_valid, regression_metrics, save_results
             train, valid = load_train_valid("hotel")
- source_code_file/3_bayesian_regression.py : 베이지안 선형 회귀 학습 코드
    python 3_bayesian_regression.py public  -> 공공 유형 하나만 실행 + 정상 작동 점검
    python 3_bayesian_regression.py         -> 10개 유형 전체 실행 + 결과 CSV + 그래프 저장
- source_data/train, source_data/test : 0_data_processing.py로 만든 건물유형별 CSV (팀 공유본과 동일)
- results/bayesian_results.csv : 유형별 성능표 (결과 CSV 열 형식 기준안)
- results/bayesian_summary.md : 결과 요약
- results/bayesian_r2_by_type.png, bayesian_band_*.png : 유형별 R2, 대표 건물 예측구간 그래프

[참고_개선실험 폴더]
팀 입력 명세 밖의 실험(건물번호, 시간대x주말, 습도 추가)입니다. 정식 비교 결과가 아닙니다.
실행해 보려면 안의 파일을 Project의 같은 위치(source_code_file, results)로 옮기면 됩니다.

필요 라이브러리: numpy, pandas, scikit-learn, matplotlib
