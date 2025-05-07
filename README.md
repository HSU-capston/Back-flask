# 자세 분석 프로세스

본 시스템은 YOLO 기반 포즈 추정기로부터 프레임별로 추출된 17개의 주요 신체 키포인트 데이터를 입력으로 받아, 볼링 자세의 품질을 정량적으로 평가한다. 분석의 목적은 프레임 간의 자세 안정성, 특정 관절 각도 유지, 신체 부위의 일관된 움직임을 기반으로 사용자의 자세를 종합 점수화하고 피드백을 제공하는 데 있다.

프레임 단위로 수집된 키포인트는 다음과 같은 네 가지 주요 지표로 가공된다:

1. 어깨 각도 편차
2. 상체 이동 거리
3. 손목의 누적 이동량
4. 발목 높이 변화 횟수

각 항목은 프레임 크기를 기준으로 정규화된 거리 또는 각도로 계산되며, 최종적으로 사용자 수준(user_level)을 고려한 평가 함수에 의해 통합 점수(Score)로 변환된다. 본 절에서는 해당 평가 지표를 수식화하여 기술한다.

## 1. 어깨 각도 차이

**정의**  
오른쪽 어깨 각도는 다음 세 점을 기준으로 정의된다.
- A = Left Shoulder  
- B = Right Shoulder  
- C = Right Elbow  

오른팔 각도는 다음과 같이 계산된다.

$$
\theta_{\text{shoulder}} = \min \left( \angle ABC, \angle CBA \right)
$$

90도 기준 편차는 다음과 같다.

$$
\Delta \theta = \left| \theta_{\text{shoulder}} - 90^\circ \right|
$$

**평균 어깨 각도 차이**

$$
\bar{\Delta \theta} = \frac{1}{N} \sum_{i=1}^{N} \Delta \theta_i
$$


## 2. 상체 중심 이동 거리

**중심 좌표 계산**  
프레임 \( t \)에서의 상체 중심은 다음과 같이 계산된다.

$$
C_t = \frac{1}{2} \left( \frac{P_{\text{LS}}^t + P_{\text{RS}}^t}{2} + \frac{P_{\text{LH}}^t + P_{\text{RH}}^t}{2} \right)
$$

**정규화된 거리 계산**  
프레임 \( t \)와 \( t-1 \) 간의 상체 이동 거리는 다음과 같다.

$$
d_t = \sqrt{ \left( \frac{x_t - x_{t-1}}{W} \right)^2 + \left( \frac{y_t - y_{t-1}}{H} \right)^2 }
$$

여기서 \( W \)는 프레임 너비, \( H \)는 프레임 높이이다.

**평균 이동 거리**

$$
\bar{d} = \frac{1}{N-1} \sum_{t=2}^{N} d_t
$$


## 3. 손목 이동 거리 누적

오른쪽 손목의 누적 이동 거리는 다음과 같다.

$$
D_{\text{wrist}} = \sum_{t=2}^{N} \sqrt{ \left( \frac{x_t - x_{t-1}}{W} \right)^2 + \left( \frac{y_t - y_{t-1}}{H} \right)^2 }
$$

단, \( (x_t, y_t) \)는 프레임 \( t \)에서의 오른쪽 손목 좌표이다.


## 4. 발목 위치 변화 횟수

**조건 정의**  
프레임 간 발목 y좌표가 다음 조건을 만족할 경우 위치 변화 이벤트로 간주한다.

$$
\Delta y_t = \left| y_t - y_{t-1} \right|
$$

$$
\text{Switch}_t =
\begin{cases}
1, & \text{if } \Delta y_t > \frac{H}{30} \text{ and } y_t \ne 0 \\
0, & \text{otherwise}
\end{cases}
$$

**총 변화 횟수**

$$
A = \sum_{t} \text{Switch}_t
$$

## [ 최종 점수 함수 및 등급 분류 ]

위 분석 지표들을 바탕으로 최종 점수는 다음 함수로 계산된다.

$$
\text{Score} = f\left( \bar{\Delta \theta}, \bar{d}, D_{\text{wrist}}, A, \text{User Level} \right)
$$

위에서 계산된 각 지표들은 최종 점수를 계산하는 함수의 입력으로 사용되며, 해당 함수에서는 종합적인 점수를 계산하고 대규모 언어 모델(LLM)을 활용하여 피드백을 생성한다

**등급 분류 기준**

$$
\text{Grade} =
\begin{cases}
\text{EXCELLENT}, & \text{if } \text{Score} \ge 80 \\
\text{GOOD}, & \text{if } 74 \le \text{Score} < 80 \\
\text{COMMON}, & \text{if } 66 \le \text{Score} < 74 \\
\text{BAD}, & \text{otherwise}
\end{cases}
$$

# Project structure

├── app.py             # Flask 애플리케이션

├── analyze.py         # 비디오 Point 기반 analyzation

├── processing.py      # 이미지/비디오 처리 및 YOLO 모델 로드

├── utils.py           # 유틸리티 함수 (비디오 회전 등)

├── uploads/           # 업로드된 파일 저장

├── results/           # 처리된 파일 저장

├── model/             # YOLO 모델 파일

└── requirements.txt   # 필요한 라이브러리 목록
