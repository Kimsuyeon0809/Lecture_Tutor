import streamlit as st
import os
import whisper
from RAG import tool_rag  # RAG.py의 질의응답 함수 임포트

# 업로드 및 결과 저장 경로 설정
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
OUTPUT_TEXT_FILE = r"C:\Users\SSU\Desktop\lecture_tutor\audio2txt.txt"

# Whisper 모델 로드
@st.cache_resource  # Streamlit 캐싱으로 Whisper 모델 반복 로드 방지
def load_whisper_model():
    return whisper.load_model("medium")  # 필요에 따라 모델 크기 변경 가능

model = load_whisper_model()

# 웹앱 제목
st.title("🎙️ 음성 인식 + RAG 기반 질의응답 시스템")

# 1. 음성 파일 업로드 및 음성 인식
st.header("📝 Step 1: 음성 파일 업로드 및 텍스트 변환")
uploaded_file = st.file_uploader("🎤 .wav 파일을 업로드하세요", type=["wav"])

if uploaded_file:
    # 업로드된 파일 저장
    uploaded_file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(uploaded_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"파일이 성공적으로 업로드되었습니다: {uploaded_file_path}")

    # 음성 인식 수행
    if st.button("음성 인식 실행"):
        try:
            with st.spinner("음성을 텍스트로 변환 중입니다..."):
                result = model.transcribe(uploaded_file_path)
                transcribed_text = result["text"]

                # 결과를 파일에 저장
                with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as file:
                    file.write(transcribed_text)

            st.success("음성 인식 완료!")
            st.text_area("🔍 변환된 텍스트:", transcribed_text, height=200)

        except Exception as e:
            st.error(f"음성 인식 중 오류 발생: {e}")

# 2. 질의응답 시스템
if os.path.exists(OUTPUT_TEXT_FILE):
    st.header("💬 Step 2: 질의응답 시스템")
    st.info("아래 입력창에 질문을 입력하세요. 변환된 텍스트 내용을 기반으로 답변합니다.")

    # 채팅 메시지 세션 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "안녕하세요! 변환된 텍스트 내용을 기반으로 질문에 답변해 드립니다."}
        ]

    # 기존 채팅 메시지 표시
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # 사용자 질문 입력
    if user_input := st.chat_input("질문을 입력하세요"):
        # 사용자 질문 저장 및 표시
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        # RAG 기반 답변 생성
        with st.spinner("답변을 생성 중입니다..."):
            try:
                response = tool_rag(user_input)  # RAG.py의 tool_rag 함수 호출
            except Exception as e:
                response = f"오류가 발생했습니다: {e}"

        # 생성된 답변 저장 및 표시
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

# & C:/Users/SSU/anaconda/envs/TUTOR/python.exe c:\Users\SSU\Desktop\basic_RAG\streamlit_test.py
# streamlit run c:\Users\SSU\Desktop\basic_RAG\streamlit_test.py