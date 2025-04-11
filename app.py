from flask import Flask, request, jsonify, send_file,  send_from_directory, url_for
import os
from processing import process_video, process_image
from connection import s3_connection
from config import BUCKET_NAME

app = Flask(__name__)
s3 = s3_connection()

# 업로드 폴더 설정
UPLOAD_FOLDER = 'uploads/'
RESULT_FOLDER = 'results/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

@app.route('/processed-files/<filename>')
def processed_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

# 업로드된 파일을 저장하고 처리하는 엔드포인트
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify(message="파일이 없습니다."), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify(message="파일 이름이 비어 있습니다."), 400
    
    # 파일 확장자 체크 (이미지 또는 비디오)
    file_extension = file.filename.split('.')[-1].lower()

    # 파일 저장 경로
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    
    try:
        # 폴더가 없으면 생성
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        
        # 파일 저장
        file.save(filename)
        print(f"파일 저장 성공: {filename}")

        # 이미지 또는 비디오 파일 처리
        if file_extension in ['mp4', 'avi', 'mov']:
            final_score, grade, guide, processed_video_path = process_video(filename)  # 비디오 처리 함수 호출
            
            # video_url = url_for('processed_file', filename=os.path.basename(processed_video_path), _external=True)
            upload_video_to_s3(processed_video_path, filename)

            response = {
                'video_url': "https://sportyup-s3.s3.ap-northeast-2.amazonaws.com/BowlingAnalyze/" + filename,
                'message1': guide,
                'message2': grade,
                'score': final_score  # 예시 점수
            }
            return jsonify(response)
        else:
            processed_image_path = process_image(filename)  # 이미지 처리 함수 호출
            return send_file(processed_image_path, as_attachment=True)  # 처리된 이미지 반환

    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return jsonify(message=f"파일 저장 중 오류 발생: {str(e)}"), 500

def upload_video_to_s3(processed_video_path, filename):
    try:
        # 비디오 파일 열기 (바이너리 모드로)
        with open(processed_video_path, 'rb') as video_file:
            # 비디오 파일을 S3에 업로드
            s3.put_object(
                Bucket=BUCKET_NAME,
                Body=video_file,
                Key="BowlingAnalyze/" + filename,  # S3에서 파일이 저장될 위치
                ContentType='video/mp4'  # MIME 타입 지정
            )
            print("비디오 파일이 성공적으로 S3에 업로드되었습니다.")
    except Exception as e:
        print(f"비디오 업로드 중 오류 발생: {str(e)}")

if __name__ == '__main__':
    # 폴더가 없으면 생성
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    if not os.path.exists(RESULT_FOLDER):
        os.makedirs(RESULT_FOLDER)
    
    # Flask 앱 실행
    app.run(debug=True, port=5000)
