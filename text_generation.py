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
        "very_very_low": 0,
        "very_low": 0.19,
        "low": 0.20,
        "slightly_low": 0.21,
        "normal_min": 0.22,
        "normal": 0.23,
        "normal_max": 0.25,
        "slightly_high": 0.26,
        "high": 0.27,
        "very_high": 1,
        "very_very_high": 2
    },
    "wrist_movement_total": {
        "very_very_low": 0,
        "very_low": 5,
        "low": 7,
        "slightly_low": 9,
        "normal_min": 10,
        "normal": 15,
        "normal_max": 16,
        "slightly_high": 16.5,
        "high": 17,
        "very_high": 28,
        "very_very_high": 30
    },
    "ankle_switch_count": {
        "very_very_low": 0,
        "very_low": 1,
        "low": 2,
        "slightly_low": 3,
        "normal_min": 4,
        "normal": 4.5,
        "normal_max": 5,
        "slightly_high": 6,
        "high": 7,
        "very_high": 8,
        "very_very_high": 9
    }
}

# 수치를 1~10 점수로 변환하는 함수
def interpret_value_to_score(value, thresholds):
    if value < thresholds["very_low"]:
        return 1
    elif value < thresholds["low"]:
        return 2
    elif value < thresholds["slightly_low"]:
        return 3
    elif value < thresholds["normal_min"]:
        return 4
    elif value < thresholds["normal"]:
        return 5
    elif value <= thresholds["normal"]:
        return 6
    elif value <= thresholds["normal_max"]:
        return 7
    elif value <= thresholds["slightly_high"]:
        return 8
    elif value <= thresholds["high"]:
        return 9
    elif value <= thresholds["very_high"]:
        return 10
    else:
        return 10

# 메인 평가 함수
def evaluate_bowling_form(avg_shoulder_angle_diff, avg_movement, wrist_movement_total, ankle_switch_count):
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

- 평균 어깨 각도 차이 (90도에서): {avg_shoulder_angle_diff}도 → 1~10 구간 중 {interpretations["shoulder_angle_diff"]}단계
- 평균 이동 거리: {avg_movement} → 1~10 구간 중 {interpretations["movement_distance"]}단계
- 손목 이동 거리 총합: {wrist_movement_total} → 1~10 구간 중 {interpretations["wrist_movement_total"]}단계
- 발목 높이 변화 이벤트 수: {ankle_switch_count} → 1~10 구간 중 {interpretations["ankle_switch_count"]}단계

※ 괄호 안 숫자는 1(가장 부족) ~ 10(가장 과한) 단계 중 몇 번째인지 나타냅니다 5나 6에 가까울 수록 높은 점수 입니다 5나 6이 아니라면 나쁜 점수 입니다 4나 7도 나쁜 점수 입니다 점수를 산정할 떄 크게 반영 하세요.

이 결과를 바탕으로 저의 볼링 자세에 대한 평가와 피드백을 주시고, 잘 된 점과 개선이 필요한 점을 알려주세요.
그리고 개선이 필요한 점이 크게 없다면, 개선이 필요한 점을 말할 때 잘하고 있지만 어떻게 더 했으면 좋겠다는 느낌으로 말해주세요.
반대로 개선이 필요한 점이 많으면 잘 된 점이 없다고 솔직하게 말해주세요. 
또한, 다음 투구는 어떻게 하면 좋을지 추천해 주세요 이 또한 개선할 점이 많이 없다면 유지하라고 짧게 말하고 다음 자세를 추천해 주세요. 마지막으로 총 평가 점수를 0부터 100 사이 숫자로 제시해 주세요.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "당신은 볼링 자세에 대한 전문가입니다. 주어진 데이터를 바탕으로 자세 평가를 "
                    "4가지 항목으로 나눠서 제공하세요: "
                    "1) 잘한 점, 2) 개선이 필요한 점, 3) 다음 투구를 위한 추천, 4) 총 평가 점수 (0~100점 사이 숫자). "
                    "이동 거리나 수치적인 언급은 자제하고 대신 '크게', '적게', '많이' 등으로 표현해 주세요. "
                    "전체 피드백은 반드시 JSON 코드블럭 내에 반환해 주세요. 예:\n"
                    "```json\n"
                    "{\"잘한점\": \"...\", \"개선점\": \"...\", \"추천\": \"...\", \"점수\": 85}\n"
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

    return result
