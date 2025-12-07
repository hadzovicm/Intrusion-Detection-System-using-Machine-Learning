import streamlit as st
import pickle
from pathlib import Path
from scipy.sparse import hstack

model = pickle.load(open("models/model.pkl", "rb"))
vectorizers = pickle.load(open("models/vectorizers.pkl", "rb"))
threshold_path = Path("models/threshold.txt")
THRESHOLD = float(threshold_path.read_text().strip()) if threshold_path.exists() else 0.0

def build_text(method, url, content):
    return f"{method.lower().strip()} {url.lower().strip()} {content.lower().strip()}"

def classify(method, url, content):
    text = build_text(method, url, content)
    word_vec = vectorizers["word"].transform([text])
    char_vec = vectorizers["char"].transform([text])
    vec = hstack([word_vec, char_vec])
    score = model.decision_function(vec)[0]
    return 1 if score >= THRESHOLD else 0

st.title("🔍 Machine Learning Intrusion Detection System")

method = st.selectbox("HTTP Method", ["GET", "POST"])
url = st.text_input("URL")
content = st.text_area("HTTP Body (optional)")

if st.button("Scan Request"):
    result = classify(method, url, content)
    if result == 1:
        st.error("🚨 Malicious Request Detected!")
    else:
        st.success("✔ Normal Request")
