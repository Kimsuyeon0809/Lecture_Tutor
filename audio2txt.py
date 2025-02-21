import whisper

# 음성 파일 경로
audio_file_path = r"C:\Users\SSU\Desktop\basic_RAG\merged_sample_audio.wav"

# 결과 저장 경로
output_text_file = r"C:\Users\SSU\Desktop\lecture_tutor\audio2txt.txt"

# Whisper 모델 로드 (medium 모델 사용)
model = whisper.load_model("medium")  # 'base', 'small', 'medium', 'large' 중 선택 가능

# 음성 인식 수행
result = model.transcribe(audio_file_path)

# 인식된 텍스트 가져오기
transcribed_text = result["text"]

# 결과 저장
with open(output_text_file, "w", encoding="utf-8") as file:
    file.write(transcribed_text)

print(f"음성 인식 결과가 {output_text_file}에 저장되었습니다.")
