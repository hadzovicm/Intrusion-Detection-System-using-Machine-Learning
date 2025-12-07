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
    vector = hstack([word_vec, char_vec])
    score = model.decision_function(vector)[0]
    return 1 if score >= THRESHOLD else 0

if __name__ == "__main__":
    while True:
        m = input("Method: ")
        if m == "q": break
        u = input("URL: ")
        c = input("Content: ")
        print("Malicious" if classify(m,u,c) else "Normal")
