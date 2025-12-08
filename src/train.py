import pickle
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    fbeta_score,
)
from sklearn.model_selection import cross_val_score
from preprocess import (
    X_train_vec,
    X_val_vec,
    X_test_vec,
    y_train,
    y_val,
    y_test,
    word_tfidf,
    char_tfidf,
)

model = LinearSVC(class_weight="balanced")
model.fit(X_train_vec, y_train)

# Sanity check: train vs validation at default threshold (0)
train_preds = model.predict(X_train_vec)
val_preds_base = model.predict(X_val_vec)
print("TRAIN ACCURACY (threshold=0):", accuracy_score(y_train, train_preds))
print("VAL ACCURACY (threshold=0):", accuracy_score(y_val, val_preds_base))
print("VAL REPORT (threshold=0):\n", classification_report(y_val, val_preds_base))
val_cm_base = confusion_matrix(y_val, val_preds_base)

# Tune threshold on the validation set only to avoid leaking test information
val_scores = model.decision_function(X_val_vec)
threshold_grid = sorted(
    set([0.0] + list(np.percentile(val_scores, np.linspace(1, 99, 80))))
)

best_thresh = 0.0
best_fbeta = -1.0
best_stats = val_cm_base.ravel().tolist()

for t in threshold_grid:
    preds = (val_scores >= t).astype(int)
    fbeta = fbeta_score(y_val, preds, beta=0.5)
    if fbeta > best_fbeta:
        best_fbeta = fbeta
        best_stats = list(confusion_matrix(y_val, preds).ravel())
        best_thresh = t

tn, fp, fn, tp = best_stats
recall = tp / (tp + fn) if (tp + fn) else 0.0
precision = tp / (tp + fp) if (tp + fp) else 0.0
print(
    f"Selected threshold={best_thresh:.4f} on validation (precision={precision:.4f}, recall={recall:.4f}, FP={fp}, FN={fn}, TP={tp}, TN={tn})"
)

# Evaluate on the untouched test set using the tuned threshold
test_scores = model.decision_function(X_test_vec)
test_preds = (test_scores >= best_thresh).astype(int)
test_cm = confusion_matrix(y_test, test_preds)
print("TEST ACCURACY:", accuracy_score(y_test, test_preds))
print("TEST CONFUSION MATRIX:\n", test_cm)
print("TEST REPORT:\n", classification_report(y_test, test_preds))

# Cross-validation on the training split to gauge stability
cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring="f1")
print(
    f"5-fold CV F1 on train split: mean={cv_scores.mean():.4f}, std={cv_scores.std():.4f}"
)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizers.pkl", "wb") as f:
    pickle.dump({"word": word_tfidf, "char": char_tfidf}, f)

with open("models/threshold.txt", "w") as f:
    f.write(str(best_thresh))

print("Saved model, vectorizer, and threshold to models/")
