from ultralytics import YOLO
import cv2
import os
from PIL import Image
import numpy as np
import analyze
import subprocess
from utils import get_video_rotation, rotate_video

# YOLO 모델 로드
model = YOLO("model/yolo11m-pose.pt")

def process_video(video_path):
    try:
        # YOLO로 비디오 파일 전체 처리
        results = model(video_path, stream=True)
        
        result_video_name = video_path.split('/')[-1].split('.')[0] + '_after.mp4'
        result_video_path = os.path.join('results', result_video_name)
        
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
       # out = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc(*'X264'), 30, (frame_width, frame_height))
        out = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

        # 키포인트 데이터를 저장할 배열
        all_keypoints_data = []

        for result in results:
            # 감지된 객체가 있는지 먼저 확인
            if len(result) == 0 or result[0].keypoints is None or result[0].keypoints.xy is None:
                # 키포인트가 없을 경우, 빈 리스트로 추가하고 넘어감
                all_keypoints_data.append([])
                annotated_frame = result.plot()
                out.write(annotated_frame)
                continue

            # 키포인트 추출
            keypoints = result[0].keypoints
            xy = keypoints.xy  # (x, y) 좌표

            # 현재 프레임에 대한 keypoints 데이터
            frame_keypoints = []

            for i in range(len(xy[0])):
                x, y = xy[0][i]
                frame_keypoints.append((x, y))

            all_keypoints_data.append(frame_keypoints)

            annotated_frame = result.plot()
            out.write(annotated_frame)
        
        out.release()
        
        rotate_result_video_path = rotate_video(result_video_path, -90)
        
        reencoded_path = rotate_result_video_path.replace('.mp4', '_web.mp4')
        reencode_to_browser_compatible(rotate_result_video_path, reencoded_path)
      
        cap.release()
        
        # 유효한 키포인트가 하나라도 있는지 확인
        has_valid_data = any(len(frame) > 0 for frame in all_keypoints_data)

        if not has_valid_data:
            final_score = 50
            grade = "BAD"
            guide_good_point = "분석 가능한 자세가 감지되지 않았습니다."
            guide_bad_point = "영상에서 사람 또는 키포인트를 인식하지 못했습니다."
            guide_recommend = "카메라 위치를 조정하거나 조명을 개선해주세요."
            
            print(f"점수: {final_score}")
            print(f"등급: {grade}")
            print(f"잘한 점: {guide_good_point}")
            print(f"부족한 점: {guide_bad_point}")
            print(f"추천: {guide_recommend}")
        else:
            # 객체가 감지된 경우에는 원래 analyze 함수 호출
            final_score, grade, guide_good_point, guide_bad_point, guide_recommend = analyze.analyze(
                all_keypoints_data, frame_width, frame_height
            )
        
        return final_score, grade, guide_good_point, guide_bad_point, guide_recommend, reencoded_path
    except Exception as e:
        print(f"비디오 처리 중 오류 발생: {str(e)}")

def process_image(image_path):
    try:
        # YOLO 모델로 이미지 처리
        results = model(image_path)
        result_image_name = image_path.split('/')[-1].split('.')[0] + '_after.' + image_path.split('.')[-1]
        result_image_path = os.path.join('results', result_image_name)

        # 결과 이미지 생성
        annotated_image = results[0].plot()
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(annotated_image_rgb)
        
        pil_image.save(result_image_path)
        return result_image_path

    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {str(e)}")

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
        print(f"ffmpeg 재인코딩 실패: {e}")
        return input_path  # 재인코딩 실패 시 원본 그대로 리턴
