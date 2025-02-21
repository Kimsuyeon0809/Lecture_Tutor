### RAG_db.py
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings # 아래 코드로 변경 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from difflib import get_close_matches
from langchain.tools import Tool # 이거 안쓰는듯 
from langchain.agents import initialize_agent, AgentType
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import requests
# 1. Data Load
# 파일 경로
output_txt = r"C:\Users\SSU\Desktop\lecture_tutor\audio2txt.txt" 

file_path = output_txt # audio2txt.py에서 생성한 txt 

# TextLoader에 인코딩 설정
loader = TextLoader(file_path, encoding='utf-8')  # 또는 확인된 인코딩으로 설정 (예: 'euc-kr', 'cp949')
data = loader.load()

# 2. text split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=100,    
    length_function = len,
)
texts = text_splitter.split_documents(data)
# Document 객체에서 텍스트 데이터 추출
texts = [doc.page_content for doc in texts]

print('나뉜 텍스트 파일 수 :', len(texts))
print(texts[0])

# 3. Embedding model 정의
embeddings_model = HuggingFaceEmbeddings(
    # model_name='jhgan/ko-sroberta-nli', # 한국어 자연어 추론에 최적화된 모델 지정 
    model_name= 'sentence-transformers/multi-qa-mpnet-base-dot-v1', #  QA 작업에 특화된 임베딩 모델로 변경 
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
)

# 4. vector store : Chroma 

db = Chroma.from_texts(
    texts, # 검색할 대상 텍스트들 
    embeddings_model, # 임베딩 모델 
    collection_name = 'history',
    persist_directory = './db', # 저장할 디렉토리 
    collection_metadata = {'hnsw:space': 'cosine'}, # 코사인 유사도 사용 
)
print("임베딩 및 데이터베이스 생성 완료.")

### 3. RAG_search.py

# 5. Retriever
# TXT 파일 검색 함수 정의
def search_txt_file(query):
    """
    텍스트 파일에서 사용자의 질문과 가장 유사한 문장을 검색.
    """
    retriever = db.as_retriever(
        search_type='mmr', # 검색의 관련성과 다양성 조정
        search_kwargs={'k': 5, 'lambda_mult': 0.15} # 관련있는 5개 문서, lambda_mult : 검색의 관련성과 다양성 조정
        )
    docs = retriever.invoke(query)

    return docs


# 6. LLM Load / ollama list -> ollama run llama3 

llm = OllamaLLM(model="llama3")

prompt_template = PromptTemplate.from_template(
    """
    Please provide answer for question from the following context. 
    - Be sure to answer in Korean.
    - If the answer to the question is not in the document, answer within the knowledge you know.
    - Answer like you teach your friend.
    ---
    Question: {question}
    ---
    Context: {context}
    """
)
chain = prompt_template | llm | StrOutputParser()


def tool_rag(question):
    """
    TXT 파일에서 먼저 검색하고, 없으면 Ollama 모델로 질문에 답변.
    """
    # 1. TXT 파일에서 검색
    txt_result = search_txt_file(question)
    if txt_result:
        prompt = (
            f"Please answer your questions based on the following information. : {txt_result}\n\question: {question}"
            f"\nFollow the rules below when answering. The most important thing is answered in Korea.\n-If the answer to the question is not in the document, answer within the knowledge you know."
            f"\n- Answer like you teach your friend.\n- Answer in Korean.")
        response = llm.invoke(prompt)
        return response

    # 2. TXT 파일에 답변이 없으면 Ollama가 직접 답변 생성
    fallback_message = "수업 내용에서 찾지 못했지만 제가 알고 있는 정보를 바탕으로 답변 드릴게요!\n"
    response = llm.invoke(f"question: {question}\nAnswer in Korean.")
    return fallback_message + response

