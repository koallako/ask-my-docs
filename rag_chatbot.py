import streamlit as st
from openai import OpenAI
import tiktoken
import faiss
import numpy as np


client = OpenAI(api_key="sk-") #일단 이렇게 두었지만, .env 파일로 따로 관리하기


def load_txt(file):
    return file.read().decode('utf-8')


def split_text(text, max_tokens=500): #텍스트 토큰화
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = encoding.decode(tokens[i:i + max_tokens])
        chunks.append(chunk)
    return chunks


def get_embedding(text): #임베딩
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding


def build_faiss_index(chunks):
    dimension = 1536  # ada-002 임베딩 차원
    index = faiss.IndexFlatL2(dimension)
    embeddings = [get_embedding(chunk) for chunk in chunks]
    index.add(np.array(embeddings).astype('float32'))
    return index, chunks, embeddings


def search_faiss(query, index, chunks, top_k=3):
    query_vec = np.array([get_embedding(query)]).astype('float32')
    distances, indices = index.search(query_vec, top_k)
    retrieved_chunks = [chunks[i] for i in indices[0] if i != -1]
    return "\n".join(retrieved_chunks)


def ask_gpt(query, context): #답변 생성
    prompt = f"""
다음 내용을 참고해서 질문에 한국어로 답변해줘.

[참고 내용]:
{context}

[질문]:
{query}

[답변]:
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def main(): #Streamlit 실행
    st.set_page_config(page_title="📄 TXT 문서 기반 RAG 챗봇", layout="wide")
    st.title("📄 TXT 문서 기반 RAG 챗봇")
    st.write("업로드한 TXT 문서를 바탕으로 질문하면 답변해드립니다!")

    uploaded_file = st.file_uploader("TXT 문서를 업로드하세요", type="txt")

    if uploaded_file:
        with st.spinner("TXT 문서 읽는 중..."):
            text = load_txt(uploaded_file)
            chunks = split_text(text)
            index, chunks, _ = build_faiss_index(chunks)
        st.success("✅ 문서 임베딩 및 인덱스 생성 완료")

        query = st.text_input("질문을 입력하세요:")
        if st.button("질문하기") and query:
            with st.spinner("답변 생성 중..."):
                context = search_faiss(query, index, chunks)
                answer = ask_gpt(query, context)
            st.markdown("### 💬 답변")
            st.write(answer)

if __name__ == "__main__":
    main()
