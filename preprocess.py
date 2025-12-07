import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# ------------------------------
# 1. Load your cleaned CSV
# ------------------------------
df = pd.read_csv("csic_database.csv")

# ------------------------------
# 2. Handle missing values
# ------------------------------

df["Method"] = df["Method"].fillna("")
df["URL"] = df["URL"].fillna("")
df["content"] = df["content"].fillna("")

# ------------------------------
# 3. Normalize text
# ------------------------------

df["Method"] = df["Method"].str.lower().str.strip()
df["URL"] = df["URL"].str.lower().str.strip()
df["content"] = df["content"].str.lower().str.strip()

# ------------------------------
# 4. Combine into one text field
# ------------------------------

df["full_request"] = df["Method"] + " " + df["URL"] + " " + df["content"]

# ------------------------------
# 5. Define X and y
# ------------------------------

X = df["full_request"]
y = df["classification"]

# ------------------------------
# 6. Train/Test Split
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# ------------------------------
# 7. TF-IDF Vectorization
# ------------------------------

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    analyzer="word"
)

X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

print("Preprocessing completed successfully!")
print("Train matrix shape:", X_train_vec.shape)
print("Test matrix shape:", X_test_vec.shape)
