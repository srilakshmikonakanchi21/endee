import streamlit as st
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("AI Document Chatbot (Endee Project)")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text()

    chunks = text.split("\n")
    embeddings = model.encode(chunks)

    st.success("Document processed!")

    query = st.text_input("Ask a question:")

    if query:
        query_embedding = model.encode([query])
        similarities = np.dot(embeddings, query_embedding.T).flatten()
        top_index = similarities.argmax()

        st.subheader("Answer:")
        st.write(chunks[top_index])
