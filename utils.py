import subprocess
import cv2

def get_video_rotation(image_path): # 회전 정보 확인 함수
    try:
        command = ['ffmpeg', '-i', image_path]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.stderr:
            for line in result.stderr.splitlines():
                if "displaymatrix: rotation" in line:
                    rotation = line.split('rotation of')[-1].strip()
                    print(f"회전 정보 있음: {float(rotation.split()[0])} degrees")
                    return float(rotation.split()[0])
        return None

    except Exception as e:
        print(f"메타데이터 추출 중 오류 발생: {str(e)}")
        return None

def rotate_video(image_path, rotation):
    cap = cv2.VideoCapture(image_path)
    if not cap.isOpened():
        print("비디오 파일을 열 수 없습니다.")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = f"{image_path.split('.')[0]}_rotated.mp4"
    
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), 20.0, (height, width))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if rotation == -90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        out.write(frame)
    
    cap.release()
    out.release()
    return output_path
