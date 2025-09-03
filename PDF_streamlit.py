import os
import streamlit as st
from openai import OpenAI
import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from io import BytesIO

# 🔑 Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# 1️⃣ Load text from PDF using LangChain's PDF loader
def extract_pdf_text(uploaded_file):
    temp_path = "temp_uploaded.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    text = "\n".join([d.page_content for d in docs])
    return text

# 2️⃣ Split text into chunks (basic splitter)
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# 3️⃣ Embed text using OpenAI
def embed_texts(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",  # light + fast
        input=texts
    )
    embeddings = [e.embedding for e in response.data]
    return np.array(embeddings, dtype=np.float32)

# 4️⃣ Build FAISS index
def build_faiss_index(chunks):
    embeddings = embed_texts(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index, embeddings

# 5️⃣ Retrieve relevant chunks
def retrieve_chunks(query, chunks, index, k=3, threshold=0.3):
    query_emb = embed_texts([query])
    distances, indices = index.search(query_emb, k)

    retrieved = []
    for score, idx in zip(distances[0], indices[0]):
        if score >= threshold:
            retrieved.append((score, chunks[idx]))
    return retrieved

# 6️⃣ Ask question with RAG
def ask_question(query, chunks, index, threshold=0.3):
    retrieved = retrieve_chunks(query, chunks, index, threshold=threshold)

    if not retrieved:
        return "⚠️ No relevant context found in the PDF."

    context = "\n".join([c for _, c in retrieved])

    prompt = f"""
    You are a helpful assistant. Use the following context to answer the question.

    Context:
    {context}

    Question: {query}
    Answer:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

# --------------------------
# 🚀 Streamlit UI
# --------------------------
st.set_page_config(page_title="📖 PDF Q&A Bot", layout="centered")

st.title("📖 PDF Question Answering Bot")
st.write("Upload a PDF and ask questions about its content!")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with st.spinner("Extracting text and creating index..."):
        pdf_text = extract_pdf_text(uploaded_file)

        if not pdf_text.strip():
            st.error("⚠️ No extractable text found in this PDF (might be scanned without OCR).")
        else:
            chunks = chunk_text(pdf_text)
            index, _ = build_faiss_index(chunks)
            st.success("✅ PDF processed successfully!")

            query = st.text_input("💡 Ask a question about the PDF:")

            if query:
                with st.spinner("Thinking..."):
                    answer = ask_question(query, chunks, index)
                st.markdown("### 📝 Answer:")
                st.write(answer)
