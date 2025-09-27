
# Hybrid Score Rule → Regression Model Trading Strategy

본 프로젝트는 **ML/DL 출력 + 전통적 지표**를 결합한 **스코어 룰 기반 전략**을  
**회귀 모델(Regression)**을 이용해 데이터 기반으로 업데이트하는 **하이브리드 트레이딩 시스템**입니다.

---

## 📊 개요
- **문제**: 전통 룰 기반 전략은 if 조건이 많아질수록 신호가 희소해지고 경직됨.
- **해결**: 
  - 지표들을 스코어화하여 결합
  - 스코어의 가중치를 회귀 모델(로지스틱/선형)로 학습 → 데이터 기반 최적화
- **핵심 구조**
  ```
  (ML/DL 출력 + 전통 지표) → 스코어 룰 → 회귀 모델 업데이트 → 최종 매매 신호
  ```

---

## ⚙️ 워크플로우
1. **피처 구성**
   - ML/DL 출력: 상승 확률 `p_ml`, 예상 수익률 `r_ml`
   - 전통 지표 점수: `s_rsi`, `s_macd`, `s_vol`, `s_atr` …

2. **초기 스코어 룰**
   ```python
   score_raw = w_ml*p_ml + w_ret*r_ml + w_rsi*s_rsi + w_macd*s_macd + w_vol*s_vol
   ```

3. **회귀 모델 업데이트**
   - 분류: 로지스틱 회귀 → `score_hat = σ(β0 + Σ βi * feature_i)`
   - 회귀: 선형/릿지/라쏘 → `score_hat = β0 + Σ βi * feature_i`

4. **실행 룰**
   - 진입: `score_hat > θ_entry` & ADX > 25
   - 포지션 사이징: `size = k * score_hat / vol`
   - 리스크 관리: 손절/익절, 일일 DD 한도, 수수료·슬리피지 반영

---

## 🧩 코드 예시
```python
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

def fit_roll(X, y, win_train=2000, win_test=200):
    preds = []
    for start in range(0, len(X)-win_train-win_test, win_test):
        tr = slice(start, start+win_train)
        te = slice(start+win_train, start+win_train+win_test)
        base = LogisticRegression(C=1.0, penalty='l2',
                                  class_weight='balanced',
                                  max_iter=1000)
        clf = CalibratedClassifierCV(base, method='isotonic', cv=3).fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:,1]  # 예측 확률 (score_hat)
        preds.append(p)
    return np.concatenate(preds)
```

---

## ✅ 체크리스트
- 데이터 누설 방지 (미래 데이터 포함 금지, 시차 처리)
- 롤링 표준화 (z-score)로 정규화
- 워크포워드 검증 필수
- 메트릭: Sharpe, MDD, HitRatio, Payoff, PnL 분포
- 장세별 성능 분해 (상승/하락/횡보)

---

## ⏱️ 봉 단위 적용
- **단타/데이**: 1~5분봉 (업데이트 주기 짧게)
- **스윙**: 30분~4시간봉 (업데이트 주기 일 단위)

---

## 📌 요약
이 시스템은 **ML/DL 모델을 팩터(스코어)로 활용**하고,  
**회귀 모델이 가중치를 최적화**하여 전략을 지속적으로 업데이트합니다.  

→ **블랙박스 위험 ↓, 해석 가능성 ↑, 실전 안정성 ↑**



