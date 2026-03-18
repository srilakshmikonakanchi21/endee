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

# Prepare job texts
job_texts = [job["title"] + " " + job["skills"] for job in jobs]

# Convert job descriptions to embeddings
job_embeddings = model.encode(job_texts)

# Search function (replacement for Endee)
def search(query_embedding, job_embeddings, job_texts):
    scores = np.dot(job_embeddings, query_embedding[0])
    top_indices = np.argsort(scores)[::-1][:3]
    results = []
    for i in top_indices:
        results.append({
            "job": jobs[i]["title"],
            "skills": jobs[i]["skills"],
            "score": float(scores[i])
        })
    return results

# Streamlit UI
st.title("AI Resume Job Matcher 🚀")

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    resume_text = ""

    for page in reader.pages:
        if page.extract_text():
            resume_text += page.extract_text()

    st.success("Resume uploaded successfully!")

    # Convert resume to embedding
    resume_embedding = model.encode([resume_text])

    # Get results
    results = search(resume_embedding, job_embeddings, job_texts)

    st.subheader("Top Job Matches:")

    for r in results:
        st.write(f"🔹 Job: {r['job']}")
        st.write(f"Skills: {r['skills']}")
        st.write(f"Match Score: {r['score']:.2f}")
        st.write("------")
