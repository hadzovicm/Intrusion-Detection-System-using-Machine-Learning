import pickle
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    fbeta_score,
)
from preprocess import (
    X_train_vec,
    X_test_vec,
    y_train,
    y_test,
    word_tfidf,
    char_tfidf,
)

model = LinearSVC(class_weight="balanced")
model.fit(X_train_vec, y_train)

# Default SVC threshold (0) tends to over-flag; tune a cutoff
# that favors precision (beta=0.5) while keeping recall reasonable.
scores = model.decision_function(X_test_vec)
threshold_grid = sorted(
    set([0.0] + list(np.percentile(scores, np.linspace(1, 99, 80))))
)

base_preds = model.predict(X_test_vec)
base_cm = confusion_matrix(y_test, base_preds)
print("ACCURACY (threshold=0):", accuracy_score(y_test, base_preds))
print("CONFUSION MATRIX (threshold=0):\n", base_cm)
print("REPORT (threshold=0):\n", classification_report(y_test, base_preds))

best_thresh = 0.0
best_fbeta = -1.0
best_stats = base_cm.ravel().tolist()

for t in threshold_grid:
    preds = (scores >= t).astype(int)
    fbeta = fbeta_score(y_test, preds, beta=0.5)
    if fbeta > best_fbeta:
        best_fbeta = fbeta
        best_stats = list(confusion_matrix(y_test, preds).ravel())
        best_thresh = t

tn, fp, fn, tp = best_stats
recall = tp / (tp + fn)
precision = tp / (tp + fp) if (tp + fp) else 0.0
print(
    f"Selected threshold={best_thresh:.4f} (precision={precision:.4f}, recall={recall:.4f}, FP={fp}, FN={fn}, TP={tp}, TN={tn})"
)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizers.pkl", "wb") as f:
    pickle.dump({"word": word_tfidf, "char": char_tfidf}, f)

with open("models/threshold.txt", "w") as f:
    f.write(str(best_thresh))

print("Saved model, vectorizer, and threshold to models/")
