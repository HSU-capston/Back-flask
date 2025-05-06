from ultralytics import YOLO
import cv2
import os
from PIL import Image
import numpy as np
import analyze
import subprocess
import tempfile
import traceback
from utils import get_video_rotation, rotate_video

# YOLO 모델 로드
model = YOLO("model/yolo11m-pose.pt")

def process_video(video_path, user_level):
    try:
        print("[INFO] YOLO 모델로 영상 분석 시작")
        results = model(video_path, stream=True)

        temp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        result_video_path = temp_out.name
        temp_out.close()

        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] 해상도: {frame_width} x {frame_height}")

        out = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

        all_keypoints_data = []

        for i, result in enumerate(results):
            print(f"[INFO] 프레임 {i} 처리 중...")
            if len(result) == 0 or result[0].keypoints is None or result[0].keypoints.xy is None:
                all_keypoints_data.append([])
                out.write(result.plot())
                continue

            xy = result[0].keypoints.xy
            frame_keypoints = [(x, y) for x, y in xy[0]]
            all_keypoints_data.append(frame_keypoints)
            out.write(result.plot())

        out.release()
        cap.release()

        print("[INFO] 영상 회전 중...")
        rotation = get_video_rotation(video_path)
        rotate_result_video_path = result_video_path
        
        if rotation is None:
            print("회전 정보가 없습니다.")
        else:
            rotate_result_video_path = rotate_video(result_video_path, rotation)

        reencoded_path = rotate_result_video_path.replace('.mp4', '_web.mp4')
        print("[INFO] 웹 호환 포맷으로 재인코딩 중...")
        reencode_to_browser_compatible(rotate_result_video_path, reencoded_path)

        # has_valid_data = any(len(frame) > 0 for frame in all_keypoints_data)
        # if not has_valid_data:
        #     print("[WARN] 유효한 키포인트 없음")
        #     return 50, "BAD", "분석 가능한 자세가 감지되지 않았습니다.", "영상에서 사람 또는 키포인트를 인식하지 못했습니다.", "카메라 위치를 조정하거나 조명을 개선해주세요.", reencoded_path

        # print("[INFO] 키포인트 분석 중...")
        # final_score, grade, guide_good_point, guide_bad_point, guide_recommend = analyze.analyze(
        #     all_keypoints_data, frame_width, frame_height
        # )
        
        # 유효한 키포인트가 하나라도 있는지 확인
        has_valid_data = any(len(frame) > 0 for frame in all_keypoints_data)

        if not has_valid_data:
            print("[WARN] 유효한 키포인트 없음")
            final_score = 0
            grade = "BAD"
            guide_good_point = "분석 가능한 자세가 감지되지 않았습니다."
            guide_bad_point = "영상에서 사람 또는 키포인트를 인식하지 못했습니다."
            guide_recommend = "카메라 위치를 조정하거나 조명을 개선해주세요."
            
            interpretations = {}
            interpretations["shoulder_angle_diff"] = 0
            interpretations["movement_distance"] = 0
            interpretations["wrist_movement_total"] = 0
            interpretations["ankle_switch_count"] = 0 
            
            print(f"점수: {final_score}")
            print(f"등급: {grade}")
            print(f"잘한 점: {guide_good_point}")
            print(f"부족한 점: {guide_bad_point}")
            print(f"추천: {guide_recommend}")
        else:
            print("[INFO] 키포인트 분석 중...")
            final_score, grade, guide_good_point, guide_bad_point, guide_recommend, interpretations = analyze.analyze(
                all_keypoints_data, frame_width, frame_height, user_level
            )

        print(f"[INFO] 분석 완료 - 점수: {final_score}, 등급: {grade}")
        return final_score, grade, guide_good_point, guide_bad_point, guide_recommend, interpretations, reencoded_path

    except Exception as e:
        print("[ERROR] 비디오 처리 중 예외 발생:")
        traceback.print_exc()
        return None, None, None, None, None, None

def process_image(image_path):
    try:
        print(f"[INFO] 이미지 처리 시작: {image_path}")
        results = model(image_path)
        result_image_name = image_path.split('/')[-1].split('.')[0] + '_after.' + image_path.split('.')[-1]
        result_image_path = os.path.join('results', result_image_name)

        annotated_image = results[0].plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(annotated_image_rgb)
        
        pil_image.save(result_image_path)
        print(f"[INFO] 이미지 저장 완료: {result_image_path}")
        return result_image_path

    except Exception as e:
        print(f"[ERROR] 이미지 처리 중 오류 발생: {str(e)}")
        traceback.print_exc()
        return None

def reencode_to_browser_compatible(input_path, output_path):
    try:
        subprocess.run([
            "ffmpeg",
            "-i", input_path,
            "-vcodec", "libx264",
            "-acodec", "aac",
            "-movflags", "+faststart",
            output_path
        ], check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg 재인코딩 실패: {e}")
        return input_path
