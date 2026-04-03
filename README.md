# Sleep & Lifestyle 기반 당뇨 예측 모델 개발

## 1. 프로젝트 개요

이 프로젝트는 **수면·생활습관·기초 생체지표를 기반으로 당뇨병 진단 여부(`diagnosed_diabetes`)를 예측**하는 모델을 개발하는 과정이다.  

1. **데이터 구조 파악과 EDA**
2. **가벼운 베이스라인 모델 구성**
3. **생활습관 기반 파생변수 설계**
4. **비침습형(screening) 모델과 혈액검사 포함형(benchmark) 모델 분리**
5. **부스팅/앙상블/Optuna 기반 고도화**
6. **경쟁형 데이터셋(train/test 분리)으로 확장**
7. **교차검증 중심의 재현성 높은 XGBoost 파이프라인 실험**

---

## 2. 문제 정의

### 핵심 타깃
- **예측 목표**: `diagnosed_diabetes` (0/1 이진 분류)

#### 비침습형 선별(Screening) 모델
- 사용 가능한 정보:
  - 나이, 성별
  - BMI, 허리-엉덩이 비율
  - 운동 시간
  - 식단 점수
  - 수면 시간
  - 스크린 타임
  - 가족력, 흡연, 음주, 혈압 등
- 목적:
  - 건강검진 이전 단계 또는 일상 데이터 기반 위험 선별

---

## 3. 사용 데이터와 개발 흐름의 축

- 기본 입력 변수: **24개**
- 주요 역할:
  - 실제 제출형 파이프라인
  - XGBoost / LightGBM / CatBoost 앙상블
  - SMOTE
  - OOF/Validation AUC 최적화

---

## 4. 설계 배경: 왜 수면·운동·스크린타임을 중요하게 봤는가

- **수면 시간**은 너무 짧거나 너무 길 때 당뇨 위험이 높아질 수 있다.
- **신체 활동**은 당뇨 위험을 낮추는 방향으로 작용한다.
- **스크린 타임 / 좌식 생활**은 독립적인 위험 요인으로 볼 수 있다.

---

## 5. 개발 과정 상세 정리

이 프로젝트의 실험은 크게 **비침습형 선별 모델 성능 확인**, **설명 가능한 파생변수 설계**, **혈액지표 포함 시 성능 변화 비교**, **competition형 고성능 파이프라인 구축**의 흐름으로 진행되었다.

### 5.1 베이스라인 구축과 변수 신호 확인
초기에는 `ColumnTransformer + OneHotEncoder + LogisticRegression` 기반의 단순 베이스라인을 먼저 구성해 데이터의 기본 난도를 확인했다. 이 단계에서 Accuracy는 높게 보였지만 양성 클래스 recall이 0에 가까워, 클래스 불균형과 단순 분류기의 한계가 분명하게 드러났다.

이후 EDA와 통계 검정을 통해 어떤 변수가 실제로 의미 있는 신호를 갖는지 확인했다. 주요 변수로는 `family_history_diabetes`, `age`, `bmi`, `waist_to_hip_ratio`, `systolic_bp`, `physical_activity_minutes_per_week`, `ldl_cholesterol`, `triglycerides`, `cholesterol_total` 등이 확인되었고, 특히 **가족력·연령·비만 관련 변수는 위험 증가 방향**, **운동량과 일부 생활습관 점수는 위험 감소 방향**의 신호를 보였다. 반면 수면 시간과 음주 변수는 단독 변수로서는 상대적으로 약한 신호를 보였다.

이 결과를 통해 이후 실험의 방향은 단순 원변수 투입보다, **생활습관과 위험요인을 조합한 파생변수 설계**로 옮겨갔다.

### 5.2 비침습형 선별 모델 실험
비침습형 실험에서는 생활습관과 기본 신체지표만으로 당뇨 위험을 어디까지 예측할 수 있는지 확인하는 데 초점을 두었다. 이를 위해 `Lifestyle_Score`, `Age_BMI_Interaction`, `Risk Flag`, `Total Risk Factors`와 같은 파생변수를 설계하고, XGBoost, Logistic Regression, soft voting, threshold tuning 등을 적용했다.

주요 결과는 다음과 같다.
- Lifestyle + Interaction 기반 XGBoost: **ROC-AUC 0.6562**
- Risk Flag 중심 구조: **ROC-AUC 0.6560**
- Ensemble + Threshold tuning: **ROC-AUC 0.6573**

즉, 파생변수와 앙상블을 적용하더라도 **비침습형 입력만으로는 성능이 대체로 AUC 0.65~0.66 수준에 수렴**했다. 이 단계는 생활습관 기반 입력이 조기 선별에는 의미가 있지만, 진단 수준의 강한 예측에는 한계가 있다는 점을 보여준 핵심 실험이었다.

### 5.3 설명 가능한 소형 피처셋과 점수화 실험
다음 단계에서는 성능만 높이는 것보다, **설명 가능성과 재현성이 높은 피처 구성**을 만드는 데 집중했다. 이를 위해 `Pulse_Pressure`, `Visceral_Fat_Proxy`, `Weighted_Lifestyle_Score`, `Obese_Inactive`, `Relative_BMI`, `Evidence_Lifestyle_Score`와 같은 근거 기반 파생변수를 추가하고, 소수 피처셋 중심의 모델 비교를 수행했다.

대표 실험 결과는 다음과 같다.
- 6개 선택 피처 CatBoost: **Final OOF AUC 0.6443**
- 9개 피처 비교 실험: Logistic **AUC 0.661**, RandomForest **0.657**, XGBoost **0.651**, CatBoost **0.659**, Voting **0.657**
- Optuna 기반 CatBoost 고도화: **Best AUC 약 0.6616**, 전체 OOF AUC 약 **0.6582**

이 결과는 두 가지 점을 시사했다. 첫째, **복잡한 부스팅 모델이 항상 단순 모델보다 우세하지는 않았다.** 둘째, 파생변수는 성능을 크게 폭발시키기보다는, **위험을 더 해석 가능하게 구조화하는 역할**이 더 컸다.

### 5.4 혈액지표 포함 여부에 따른 성능 차이 확인
전체 실험을 통틀어 가장 중요한 비교 중 하나는, **비침습형 입력과 임상 검사값 포함 입력의 성능 격차**를 확인한 부분이었다. 생활습관과 기본 생체지표만 사용할 때는 성능이 AUC 0.66 전후에 머물렀지만, 혈당 관련 검사가 추가되면 성능이 크게 상승했다.

대표적으로,
- `glucose_fasting` 추가 시: **AUC 약 0.80**
- `glucose_fasting + insulin_level + hba1c` 포함 시: **AUC 약 0.94**

즉, 생활습관 기반 모델은 **위험 선별용**, 혈액지표 포함 모델은 **진단 보조용**으로 문제를 분리해서 보는 것이 타당하다는 결론을 얻었다.

### 5.5 Competition 데이터 기반 고성능 파이프라인 실험
이 단계에서는 범주형 인코딩, 학습셋 기준 SMOTE, XGBoost·LightGBM·CatBoost 부스팅 모델 비교, soft voting 앙상블을 수행했다.

주요 성능은 다음과 같다.
- XGBoost Validation AUC: **0.7145**
- LightGBM Validation AUC: **0.7217**
- CatBoost Validation AUC: **0.7156**
- Soft Voting Ensemble Validation AUC: **0.72485**
- Soft Voting Ensemble Validation Accuracy: **0.66816**

여기서는 단일 모델보다 **부스팅 앙상블이 더 안정적**이었고, 특히 LightGBM과 CatBoost가 상대적으로 강한 성능을 보였다. 다만 submission 생성 단계에서는 `real_test_X` 관련 오류와 `pd` 미정의 문제가 있어, 학습 자체는 성공했지만 최종 제출 파이프라인은 추가 정리가 필요한 상태였다.

### 5.6 교차검증 중심 XGBoost 고도화
마지막 단계에서는 competition 실험을 더 발전시켜, `RepeatedStratifiedKFold(5x2)` 기반의 재현성 높은 XGBoost 파이프라인을 구성했다. 데이터셋을 결합해 공통 컬럼을 맞추고, factorize, count encoding, 조기 종료 기반 대규모 트리 학습을 적용해 OOF 중심 평가 체계를 구축했다.

이 실험의 결과는 다음과 같다.
- **OOF AUC 0.72949**
- **Train score 0.74674**

이는 전체 실험 중 가장 높은 수준의 일반화 성능이었으며, 업로드된 실험들 가운데 **가장 실전 제출형에 가까운 최종 파이프라인**으로 볼 수 있다.

---

## 6. 전체 실험 결과를 한 줄로 요약하면

- **비침습형 선별 모델은 생활습관·기초지표만으로 AUC 0.64~0.66 수준의 위험 선별 성능을 보였고, 혈당·HbA1c·인슐린 같은 임상 지표를 포함하면 AUC 0.80~0.94까지 크게 향상되었으며, competition 세팅에서는 부스팅 앙상블과 교차검증 기반 XGBoost가 최고 성능(Validation AUC 0.72485, OOF AUC 0.72949)을 기록했다.**

---

## 7. 이 프로젝트에서 실제로 배운 것

### 7.1 생활습관은 분명 의미가 있지만, 단독으로는 한계가 있다
운동, 식단, 스크린타임, 수면은 분명 신호를 주지만  
당뇨 진단을 강하게 결정짓는 정보는 결국 혈당/당화혈색소 쪽이다.

### 7.2 파생변수는 “성능 폭발”보다 “설명력 향상”에 더 큰 기여를 했다
- Lifestyle Score
- Risk Flag
- Total Risk Factors
- Evidence Lifestyle Score  

### 7.3 누설(leakage) 관리가 매우 중요하다
다음 변수는 실전 예측에서 매우 조심해야 한다.
- `diabetes_stage`
- `diabetes_risk_score`
- 경우에 따라 `hba1c`, `glucose_fasting`, `insulin_level`도  
  screening 목적에서는 사실상 정답에 너무 가까운 변수일 수 있다

### 7.4 복잡한 모델이 항상 이기는 것은 아니다
비침습형 setting에서는 Logistic Regression이 의외로 AUC 기준 가장 안정적인 구간이 있었다.  
즉, 문제 정의와 피처 구성이 바뀌면 최적 모델도 바뀐다.

---

## 8. 현재 기준 추천 모델 전략

### 생활습관 기반 조기 선별 모델
#### 변수
- age
- bmi
- waist_to_hip_ratio
- physical_activity_minutes_per_week
- diet_score
- sleep_hours_per_day
- screen_time_hours_per_day
- smoking_status
- family_history_diabetes
- systolic_bp / diastolic_bp
- Evidence_Lifestyle_Score 또는 Weighted_Lifestyle_Score

#### 추천 모델
1. **Logistic Regression**
   - 해석성 높음
   - 배포 쉬움
2. **CatBoost**
   - 범주형 처리 강점
   - 소규모/혼합형 구조화 데이터에 적합

#### 기대 성능
- ROC-AUC: **0.66 전후**

---

## 9. 추후 개선 포인트

1. **목적별 데이터셋 분리**
   - screening 전용
   - diagnostic 전용
   - competition 전용

2. **feature leakage 규칙 명문화**
   - 어떤 변수는 screening에서 금지할지 문서화

3. **threshold optimization 체계화**
   - recall 우선 모델
   - precision 우선 모델
   - balanced model

4. **calibration 추가**
   - 실제 “위험도”를 보여주는 모델이라면 확률 calibration이 중요

5. **explainability 보강**
   - SHAP summary
   - 개별 환자 risk factor explanation

---

## 10. 최종 결론

- 생활습관 기반 비침습형 모델은 **설명 가능하고 선별용으로 의미가 있다**
- 하지만 성능은 대체로 **AUC 0.67 전후**에 머문다
- 공복혈당, HbA1c, 인슐린을 포함하면 성능은 **AUC 0.80~0.94**까지 상승한다
- 대규모 competition 세팅에서는 **부스팅 앙상블과 OOF 기반 XGBoost**가 가장 강력했다
- 따라서 앞으로는  
  **“생활습관 선별 모델”과 “임상 진단 보조 모델”을 별도 제품으로 분리 설계하는 것**이 가장 합리적이다

---

## 11. 참고 자료

### 업로드 자료 기반 참고
- Sleep Health and Lifestyle Dataset 관련 회의 자료
- 수면/운동/스크린타임과 당뇨 위험 관련 문헌 정리 자료

### 문헌 메모
- Shan Z, Ma H, Xie M, et al. *Sleep duration and risk of type 2 diabetes: a meta-analysis of prospective studies.* Diabetes Care. 2015.
- Aune D, Norat T, Leitzmann M, et al. *Physical activity and the risk of type 2 diabetes: a systematic review and dose-response meta-analysis.* Eur J Epidemiol. 2015.
- Grøntved A, Hu FB. *Television Viewing and Risk of Type 2 Diabetes, Cardiovascular Disease, and All-Cause Mortality: A Meta-analysis.* JAMA. 2011.