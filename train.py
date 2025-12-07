import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from preprocess import X_train_vec, X_test_vec, y_train, y_test, tfidf

# ------------------------------
# Train the model
# ------------------------------

model = LogisticRegression(max_iter=500)
model.fit(X_train_vec, y_train)

# ------------------------------
# Evaluate
# ------------------------------

y_pred = model.predict(X_test_vec)

print("ACCURACY:", accuracy_score(y_test, y_pred))
print("\nCONFUSION MATRIX:\n", confusion_matrix(y_test, y_pred))
print("\nREPORT:\n", classification_report(y_test, y_pred))

# ------------------------------
# Save model + vectorizer
# ------------------------------

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("\nModel and TF-IDF vectorizer saved successfully!")
