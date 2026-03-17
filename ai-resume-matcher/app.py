import streamlit as st
import json
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import endee

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load jobs
with open('ai-resume-matcher/jobs.json', 'r') as f:
    jobs = json.load(f)

# Prepare job data
job_texts = [job["title"] + " " + job["skills"] for job in jobs]

# Convert to embeddings
job_embeddings = model.encode(job_texts)

# Create Endee DB
db = endee.VectorDB()
db.add(job_embeddings.tolist(), job_texts)

# Streamlit UI
st.title("AI Resume Job Matcher using Endee 🚀")

uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

if uploaded_file:
    reader = PdfReader(uploaded_file)
    resume_text = ""

    for page in reader.pages:
        if page.extract_text():
            resume_text += page.extract_text()

    st.success("Resume uploaded successfully!")

    # Convert resume to embedding
    resume_embedding = model.encode([resume_text]).tolist()

    # Search using Endee
    results = db.search(resume_embedding, top_k=3)

    st.subheader("Top Job Matches:")

    for i, res in enumerate(results):
        st.write(f"Match {i+1}: {res}")
        st.write("------")
