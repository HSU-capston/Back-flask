import os
import json
from openai import OpenAI
import config

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = config.GPT_KEY  # 보안 중요!

# 각 지표별 평가 기준 (10단계 구간)
CRITERIA = {
    "shoulder_angle_diff": {
        "very_very_low": 0,
        "very_low": 1,
        "low": 2,
        "slightly_low": 15,
        "normal_min": 18,
        "normal": 20,
        "normal_max": 24,
        "slightly_high": 30,
        "high": 40,
        "very_high": 45,
        "very_very_high": 50
    },
    "movement_distance": {
        "very_very_low": 0.002,
        "very_low": 0.0025,
        "low": 0.003,
        "slightly_low": 0.0035,
        "normal_min": 0.0043,
        "normal": 0.0044,
        "normal_max": 0.0046,
        "slightly_high": 0.0055,
        "high": 0.0075,
        "very_high": 0.008,
        "very_very_high": 0.01
    },
    "wrist_movement_total": {
        "very_very_low": 0,
        "very_low": 3,
        "low": 4,
        "slightly_low": 5,
        "normal_min": 9,
        "normal": 15,
        "normal_max": 16,
        "slightly_high": 18,
        "high": 20,
        "very_high": 30,
        "very_very_high": 35
    },
    "ankle_switch_count": {
        "very_very_low": 0,
        "very_low": 1,
        "low": 2,
        "slightly_low": 3,
        "normal_min": 3,
        "normal": 4,
        "normal_max": 5,
        "slightly_high": 6,
        "high": 7,
        "very_high": 8,
        "very_very_high": 9
    }
}

def interpret_value_to_score(value, thresholds):
    # 기준 키 순서 (11개 지점 → 11개 점수 구간, 0~10점)
    keys_ordered = [
        "very_very_low", "very_low", "low", "slightly_low",
        "normal_min", "normal", "normal_max", "slightly_high",
        "high", "very_high", "very_very_high"
    ]

    values = [thresholds[k] for k in keys_ordered]

    # 0~10점 구간 매핑
    for i in range(10):
        if values[i] <= value < values[i + 1]:
            return i  # 0~9점
    return 10  # 최상단 이상은 10점


# 메인 평가 함수
def evaluate_bowling_form(avg_shoulder_angle_diff, avg_movement, wrist_movement_total, ankle_switch_count, user_level):
    client = OpenAI()

    # 입력 값 정리
    values = {
        "shoulder_angle_diff": avg_shoulder_angle_diff,
        "movement_distance": avg_movement,
        "wrist_movement_total": wrist_movement_total,
        "ankle_switch_count": ankle_switch_count
    }

     # 점수 매기기
    interpretations = {}
    for key, value in values.items():
        thresholds = CRITERIA[key]
        score = interpret_value_to_score(value, thresholds)
        interpretations[key] = score

    # 프롬프트 작성
    prompt = f"""
    제가 볼링 자세 평가를 완료했습니다. 결과는 아래와 같습니다:

    - 평균 어깨 각도 차이 (90도에서): {avg_shoulder_angle_diff}도 → 0~10점 중 {interpretations["shoulder_angle_diff"]}점
    - 평균 이동 거리: {avg_movement} → 0~10점 중 {interpretations["movement_distance"]}점
    - 손목 이동 거리 총합: {wrist_movement_total} → 0~10점 중 {interpretations["wrist_movement_total"]}점
    - 발목 높이 변화 이벤트 수: {ankle_switch_count} → 0~10점 중 {interpretations["ankle_switch_count"]}점

    ※ 점수는 0점(가장 부족하거나 과한 상태) ~ 10점(가장 부족하거나 과한 상태)까지이며,  
    5점이 가장 이상적인 자세를 의미합니다.  
    5점에서 멀어질수록 개선이 필요한 점수이며, 0~3점 또는 7~10점은 많이 개선이 필요한 상태입니다.  
    4점 또는 6점은 약간의 개선이 필요합니다.

    유저의 현재 실력은 {user_level}입니다.  
    - BEGINNER는 4~6점도 좋은 점수로 간주될 수 있습니다.  
    - ADVANCED는 반드시 5점에 가까운 점수를 지향해야 합니다.

    이 평가 결과를 바탕으로 저의 볼링 자세에 대해 아래 내용을 포함하여 JSON 형식으로 평가해 주세요:

    1. 잘한 점  
    2. 개선이 필요한 점  
    3. 다음 투구에 대한 짧은 추천  
    4. 총 점수 (0~100점 사이의 숫자)
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 볼링 자세에 대한 전문가입니다. 주어진 데이터를 바탕으로 다음 4가지 항목으로 자세를 평가하세요:\n"
                    "1) 잘한 점, 2) 개선이 필요한 점, 3) 다음 투구를 위한 추천, 4) 총 평가 점수 (0~100점 사이 숫자).\n\n"
                    "※ 점수는 0~10점 기준이며, 5점이 가장 이상적인 자세입니다. 5점에서 멀어질수록 개선이 필요합니다.\n"
                    "BEGINNER의 경우 약간 넓게 해석하고, ADVANCED의 경우 더 엄격히 평가하세요.\n"
                    "가능하면 수치나 숫자는 직접 언급하지 말고 '많이', '적게', '부드럽게' 등으로 표현하세요.\n"
                    "전체 결과는 반드시 JSON 코드블럭 내에 다음과 같은 형식으로 제공해 주세요:\n"
                    "```json\n"
                    "{ \"잘한점\": \"...\", \"개선점\": \"...\", \"추천\": \"...\", \"점수\": 85 }\n"
                    "```"
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    # 결과 파싱
    content = response.choices[0].message.content

    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {
            "잘한점": "결과를 파싱하는 데 문제가 발생했습니다.",
            "개선점": "형식이 올바르지 않아 개선점을 가져올 수 없습니다.",
            "추천": "다음 투구를 위한 추천을 확인할 수 없습니다.",
            "점수": 50
        }
        
    #print("해석된 점수:", interpretations)
    return result,  interpretations
