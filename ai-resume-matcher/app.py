import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load jobs
with open('ai-resume-matcher/jobs.json', 'r') as f:
    jobs = json.load(f)

job_texts = [job["title"] + " " + job["skills"] for job in jobs]
job_embeddings = model.encode(job_texts)

# Search function
def search(query_embedding):
    scores = np.dot(job_embeddings, query_embedding[0])
    top = np.argsort(scores)[::-1][:3]

    results = []
    for i in top:
        results.append({
            "title": jobs[i]["title"],
            "skills": jobs[i]["skills"],
            "score": float(scores[i])
        })
    return results

# UI
st.title("AI Resume Job Matcher 🚀")

uploaded_file = st.file_uploader("Upload Resume", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    resume_embedding = model.encode([text])

    results = search(resume_embedding)

    st.subheader("Top Matches")

    for r in results:
        st.write(f"Job: {r['title']}")
        st.write(f"Skills: {r['skills']}")
        st.write(f"Score: {r['score']:.2f}")
        st.write("------")
