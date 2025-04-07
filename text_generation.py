from openai import OpenAI
import os
os.environ["OPENAI_API_KEY"] = "키 값 넣으센"

def evaluate_bowling_form(avg_shoulder_angle_diff, avg_movement, wrist_movement_total, ankle_switch_count):

    # GPT로 평가 요청
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "당신은 볼링 자세에 대한 전문가입니다. 주어진 데이터를 바탕으로 자세 평가를 제공하고 개선 방안을 제시합니다. 이동 거리나 수치적인 언급 자제하고 대신 크게 적게 많이 이런식으로 말해주고 자세와 투구의 강도 및 정확성에 대해 설명해주세요. 사용자에게 자세에 대한 피드백을 제공해 주세요."
            },
            {
                "role": "user",
                "content": f"""
                    제가 볼링 자세 평가를 완료했습니다. 결과는 아래와 같습니다:

                    - **평균 어깨 각도 차이 (90도에서)**: {avg_shoulder_angle_diff}도
                    - **평균 이동 거리**: {avg_movement}
                    - **손목 이동 거리 총합**: {wrist_movement_total}
                    - **발목 높이 변화 이벤트 수**: {ankle_switch_count}

                    이 결과를 바탕으로 저의 볼링 자세에 대한 평가와 피드백을 주시고, 잘 된 점과 개선이 필요한 점을 알려주세요. 또한, 다음 투구는 어떻게 하면 좋을지 추천해 주세요.
                """
            }
        ]
    )

    # GPT 응답 처리
    result = response.choices[0].message.content
    return result
