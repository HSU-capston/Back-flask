from openai import OpenAI
import os

os.environ["OPENAI_API_KEY"] = "키 값 넣으센"

def evaluate_bowling_form(avg_shoulder_angle_diff, avg_movement, wrist_movement_total, ankle_switch_count):

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "당신은 볼링 자세에 대한 전문가입니다. 주어진 데이터를 바탕으로 자세 평가를 3가지 항목으로 나눠서 제공하세요: 1) 잘한 점 2) 개선이 필요한 점 3) 다음 투구를 위한 추천. 이동 거리나 수치적인 언급은 자제하고 대신 '크게', '적게', '많이' 등으로 표현해 주세요. 전체 피드백은 JSON 형식으로 제공해 주세요. 예: {\"잘한점\": \"...\", \"개선점\": \"...\", \"추천\": \"...\"}"
            },
            {
                "role": "user",
                "content": f"""
                    제가 볼링 자세 평가를 완료했습니다. 결과는 아래와 같습니다:

                    - 평균 어깨 각도 차이 (90도에서): {avg_shoulder_angle_diff}도
                    - 평균 이동 거리: {avg_movement}
                    - 손목 이동 거리 총합: {wrist_movement_total}
                    - 발목 높이 변화 이벤트 수: {ankle_switch_count}

                    이 결과를 바탕으로 저의 볼링 자세에 대한 평가와 피드백을 주시고, 잘 된 점과 개선이 필요한 점을 알려주세요. 또한, 다음 투구는 어떻게 하면 좋을지 추천해 주세요.
                """
            }
        ]
    )

    # GPT 응답 처리
    import json
    content = response.choices[0].message.content

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {
            "잘한점": "결과를 파싱하는 데 문제가 발생했습니다.",
            "개선점": "형식이 올바르지 않아 개선점을 가져올 수 없습니다.",
            "추천": "다음 투구를 위한 추천을 확인할 수 없습니다."
        }

    return result
