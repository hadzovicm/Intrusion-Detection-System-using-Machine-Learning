import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/csic_database.csv")

# Add a few high-confidence normal requests that were being misclassified.
extra_normal = pd.DataFrame([
    ["get", "http://localhost:8080/tienda1/publico/buscar.jsp?search=camara", "", 0],
    ["get", "http://localhost:8080/tienda1/publico/buscar.jsp?search=camara.", "", 0],
    ["get", "http://localhost:8080/tienda1/publico/buscar.jsp?search=televisor+4k", "", 0],
    ["get", "http://localhost:8080/tienda1/publico/buscar.jsp?search=objetivo+canon", "", 0],
    ["post", "http://localhost:8080/tienda1/publico/autenticar.jsp", "modo=entrar&login=choong&pwd=d1se3cion.", 0],
    ["post", "http://localhost:8080/tienda1/publico/autenticar.jsp", "modo=entrar&login=cliente&pwd=segura123", 0],
], columns=["Method", "URL", "content", "classification"])

# Add a few explicit XSS payloads as malicious to help the model see script-like patterns.
extra_malicious = pd.DataFrame([
    ["get", "http://localhost:8080/tienda1/publico/buscar.jsp?search=<script>alert(1)</script>", "", 1],
    ["get", "http://localhost:8080/tienda1/publico/buscar.jsp?search=%3Cscript%3Ealert(1)%3C/script%3E", "", 1],
    ["get", "http://localhost:8080/tienda1/publico/buscar.jsp?search=<img src=x onerror=alert(1)>", "", 1],
], columns=["Method", "URL", "content", "classification"])

df = pd.concat([df, extra_normal, extra_malicious], ignore_index=True)

df["Method"] = df["Method"].fillna("").str.lower().str.strip()
df["URL"] = df["URL"].fillna("").str.lower().str.strip()
df["content"] = df["content"].fillna("").str.lower().str.strip()

df["full_request"] = df["Method"] + " " + df["URL"] + " " + df["content"]

X = df["full_request"]
y = df["classification"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

word_tfidf = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 4),
    token_pattern=r"(?u)\b[\w\-/+=@.]+|\S",
    sublinear_tf=True
)

char_tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 6),
    min_df=2,
    max_features=50000,
    sublinear_tf=True
)

X_train_word = word_tfidf.fit_transform(X_train)
X_train_char = char_tfidf.fit_transform(X_train)
X_train_vec = hstack([X_train_word, X_train_char])

X_test_word = word_tfidf.transform(X_test)
X_test_char = char_tfidf.transform(X_test)
X_test_vec = hstack([X_test_word, X_test_char])

if __name__ == "__main__":
    print("Preprocessing complete!")
    print("Train shape:", X_train_vec.shape)
    print("Test shape:", X_test_vec.shape)
