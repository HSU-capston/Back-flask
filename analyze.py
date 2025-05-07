import math
import text_generation

KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", 
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

def analyze(all_keypoints_data, frame_width, frame_height, user_level):
    shoulder_angles = []  # 오른쪽 어깨 각도 차이
    movements = []  # 이동 거리
    wrist_movement_total = 0  # 오른쪽 손목의 누적 이동 거리
    ankle_switch_count = 0  # 발목 높이가 바뀐 시점의 수
     
    prev_right_wrist = None  # 이전 오른쪽 손목 좌표
    prev_left_ankle_y = None  # 이전 왼쪽 발목 y 좌표
    prev_right_ankle_y = None  # 이전 오른쪽 발목 y 좌표
    prev_keypoints = None # 이전 키 포인트
    
    valid_face_count = 0
    ear_count = 0
    total_frames = len(all_keypoints_data)
    
    for frame_idx, keypoints_data in enumerate(all_keypoints_data):
        #print(f"\n Frame [{frame_idx + 1}]:")
        
        # 1. 오른쪽 어깨 각도 차이 계산
        right_shoulder_angle_diff = calculate_shoulder_angle_diff(keypoints_data)
        if right_shoulder_angle_diff:
            shoulder_angles.append(right_shoulder_angle_diff)
        
        # 2. 상체 중심 이동 거리 계산 (프레임 간 변화 기반)
        if prev_keypoints:
            movement = calculate_upperbody_movement(prev_keypoints, keypoints_data, frame_width, frame_height)
            if movement:
                movements.append(movement)
        prev_keypoints = keypoints_data
        
        # 3. 오른쪽 손목 이동 거리 누적 (상대적인 이동)
        right_wrist = keypoints_data[KEYPOINT_NAMES.index("Right Wrist")]
        if prev_right_wrist:
            wrist_movement_total += calculate_distance(prev_right_wrist, right_wrist, frame_width, frame_height)
        prev_right_wrist = right_wrist
        
        # 4. 발목 높이가 바뀌는 시점 계산 (y좌표 변화 기준)
        left_ankle_y = keypoints_data[KEYPOINT_NAMES.index("Left Ankle")][1]
        right_ankle_y = keypoints_data[KEYPOINT_NAMES.index("Right Ankle")][1]
        
        if frame_idx % 10 == 0:
            # 좌표가 0이 아닌 경우에만 계산
            if left_ankle_y != 0 and prev_left_ankle_y != 0 and prev_left_ankle_y is not None and abs(left_ankle_y - prev_left_ankle_y) > frame_height / 30:
                ankle_switch_count += 1

            if right_ankle_y != 0 and prev_right_ankle_y != 0 and prev_right_ankle_y is not None and abs(right_ankle_y - prev_right_ankle_y) > frame_height / 30:
                ankle_switch_count += 1

            # 이전 발목 y 좌표 업데이트
            prev_left_ankle_y = left_ankle_y
            prev_right_ankle_y = right_ankle_y
        
        # 동영상 유효 검사    
        nose = keypoints_data[KEYPOINT_NAMES.index("Nose")]
        left_eye = keypoints_data[KEYPOINT_NAMES.index("Left Eye")]
        right_eye = keypoints_data[KEYPOINT_NAMES.index("Right Eye")]
        left_ear = keypoints_data[KEYPOINT_NAMES.index("Left Ear")]
        right_ear = keypoints_data[KEYPOINT_NAMES.index("Right Ear")]
        
        # 얼굴이 탐지된 프레임 수
        if not is_invalid_point(nose) and not is_invalid_point(left_eye) and not is_invalid_point(right_eye):
            valid_face_count += 1

        # 귀가 탐지된 프레임 수
        if not is_invalid_point(left_ear) or not is_invalid_point(right_ear):
            ear_count += 1
    
    face_ratio = valid_face_count / total_frames
    ear_ratio = ear_count / total_frames

    # 조건에 따라 리턴
    if face_ratio > 0.7 or (ear_ratio < 0.7):
        final_score = 0
        grade = "BAD"
        guide_good_point = "분석 가능한 자세가 감지되지 않았습니다."
        guide_bad_point = "영상에서 사람 또는 키포인트를 인식하지 못했습니다."
        guide_recommend = "카메라 위치를 조정하거나 조명을 개선해주세요."
        
        print(f"\n점수: {final_score}")
        print(f"등급: {grade}")
        print(f"잘한 점: {guide_good_point}")
        print(f"부족한 점: {guide_bad_point}")
        print(f"추천: {guide_recommend}")
        
        interpretations = {}
        
        interpretations["shoulder_angle_diff"] = 0
        interpretations["movement_distance"] = 0
        interpretations["wrist_movement_total"] = 0
        interpretations["ankle_switch_count"] = 0 
        
        return final_score, grade, guide_good_point, guide_bad_point, guide_recommend, interpretations
    
    # 평균 계산
    avg_shoulder_angle_diff = sum(shoulder_angles) / len(shoulder_angles) if shoulder_angles else 0
    avg_movement = sum(movements) / len(movements) if movements else 0

    
    print(f"\n frame_width: {frame_width} frame_height: {frame_height}")
    print(f"Average Shoulder Angle Difference from 90 degrees: {avg_shoulder_angle_diff} degrees")
    print(f"Average Movement Distance: {avg_movement}")
    print(f"Total Wrist Movement Distance: {wrist_movement_total}")
    print(f"Total Ankle Height Change Events: {ankle_switch_count}")
    
    guide,  interpretations = text_generation.evaluate_bowling_form(avg_shoulder_angle_diff, avg_movement, wrist_movement_total, ankle_switch_count, user_level)
    guide_good_point = guide["잘한점"]
    guide_bad_point = guide["개선점"]
    guide_recommend = guide["추천"]
    final_score = guide["점수"]
    
    shoulder_angle_diff_score = interpretations["shoulder_angle_diff"]
    movement_distance_score = interpretations["movement_distance"]
    wrist_movement_total_score = interpretations["wrist_movement_total"]
    ankle_switch_count_score = interpretations["ankle_switch_count"]
    
    print(f" - 어깨 각도 차이 점수         : {shoulder_angle_diff_score}\n")
    print(f" - 이동 거리 점수              : {movement_distance_score}\n")
    print(f" - 손목 움직임 총합 점수       : {wrist_movement_total_score}\n")
    print(f" - 발목 교차 횟수 점수         : {ankle_switch_count_score}\n")
    
    print(f"\n점수: {final_score}")

    # 점수에 따라 등급 평가
    if final_score >= 80:
        grade = "EXCELLENT"
    elif final_score >= 74:
        grade = "GOOD"
    elif final_score >= 66:
        grade = "COMMON"
    else:
        grade = "BAD"

    print(f"등급: {grade}")
    
    print(f"잘한 점: {guide_good_point}")
    print(f"부족한 점: {guide_bad_point}")
    print(f"추천: {guide_recommend}")
    
    return final_score, grade, guide_good_point, guide_bad_point, guide_recommend, interpretations


# 오른쪽 어깨 각도 차이 계산 함수
def calculate_shoulder_angle_diff(keypoints_data):
    left_shoulder = keypoints_data[KEYPOINT_NAMES.index("Left Shoulder")]
    right_shoulder = keypoints_data[KEYPOINT_NAMES.index("Right Shoulder")]
    right_elbow = keypoints_data[KEYPOINT_NAMES.index("Right Elbow")]   
    
    # 좌표가 (0, 0)일 경우 계산을 건너뜁니다.
    if is_invalid_point(left_shoulder) or is_invalid_point(right_shoulder) or is_invalid_point(right_elbow):
        return 0
    
    angle = get_smallest_angle(left_shoulder, right_shoulder, right_elbow)
    angle_diff_from_90 = abs(angle - 90)  # 90도에서 차이 계산
    return angle_diff_from_90

def calculate_upperbody_movement(prev_keypoints, curr_keypoints, frame_width, frame_height):
    def center(p1, p2):
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def is_valid(p1, p2):
        return not (is_invalid_point(p1) or is_invalid_point(p2))

    L_S_prev = prev_keypoints[KEYPOINT_NAMES.index("Left Shoulder")]
    R_S_prev = prev_keypoints[KEYPOINT_NAMES.index("Right Shoulder")]
    L_H_prev = prev_keypoints[KEYPOINT_NAMES.index("Left Hip")]
    R_H_prev = prev_keypoints[KEYPOINT_NAMES.index("Right Hip")]

    L_S_curr = curr_keypoints[KEYPOINT_NAMES.index("Left Shoulder")]
    R_S_curr = curr_keypoints[KEYPOINT_NAMES.index("Right Shoulder")]
    L_H_curr = curr_keypoints[KEYPOINT_NAMES.index("Left Hip")]
    R_H_curr = curr_keypoints[KEYPOINT_NAMES.index("Right Hip")]

    if not all([
        is_valid(L_S_prev, R_S_prev),
        is_valid(L_H_prev, R_H_prev),
        is_valid(L_S_curr, R_S_curr),
        is_valid(L_H_curr, R_H_curr)
    ]):
        return 0

    prev_center = center(center(L_S_prev, R_S_prev), center(L_H_prev, R_H_prev))
    curr_center = center(center(L_S_curr, R_S_curr), center(L_H_curr, R_H_curr))

    return calculate_distance(prev_center, curr_center, frame_width, frame_height)


# 두 점 간의 거리 계산 함수 (상대적인 거리)
def calculate_distance(p1, p2, frame_width=None, frame_height=None):
    if frame_width and frame_height:
        # 상대적인 비율로 거리 계산 (프레임 크기를 기준으로)
        x1, y1 = p1
        x2, y2 = p2
        relative_distance = math.sqrt(((x2 - x1) / frame_width) ** 2 + ((y2 - y1) / frame_height) ** 2)
        return relative_distance
    else:
        # 절대적인 거리 계산
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

# 각도를 계산하는 함수 (작은 각도를 리턴)
def get_smallest_angle(A, B, C):
    angle_ABC = calculate_angle(A, B, C)  # A-B-C 각도
    angle_ACB = calculate_angle(C, B, A)  # C-B-A 각도
    
    return min(angle_ABC, angle_ACB)

# 두 벡터 간의 각도를 계산하는 함수
def calculate_angle(A, B, C):
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])
    
    mag_BA = math.sqrt(BA[0]**2 + BA[1]**2)  # 벡터의 크기 계산
    mag_BC = math.sqrt(BC[0]**2 + BC[1]**2)
    
    dot_product = BA[0] * BC[0] + BA[1] * BC[1]  # 벡터의 내적 계산
    cos_theta = dot_product / (mag_BA * mag_BC)  # 두 벡터 간의 코사인 각도 계산
    
    angle_rad = math.acos(cos_theta)  # 코사인 값을 아크코사인으로 변환하여 각도를 계산 (라디안)
    angle_deg = math.degrees(angle_rad)  # 라디안을 도로 변환
    
    return angle_deg

# 유효하지 않은 좌표인지 체크하는 함수 (0, 0 인지 확인)
def is_invalid_point(point):
    x, y = point
    return x == 0 and y == 0

# 키포인트 출력 함수
def print_keypoints(keypoints_data):
    for i in range(17):  # 17개 키포인트 전부를 출력
        x, y = keypoints_data[i]  # 현재 키포인트의 좌표
        if x == 0 and y == 0:
            coordinates_text = "None"
        else:
            coordinates_text = f"({int(x)}, {int(y)})"
        print(f"{KEYPOINT_NAMES[i]}: {coordinates_text}")
