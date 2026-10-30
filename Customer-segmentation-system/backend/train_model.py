"""Customer segmentation training pipeline.

Steps
1. Clean transactions and build an RFM table (Recency, Frequency, Monetary).
2. Log-transform skewed features and standardise them.
3. Pick the number of segments with KMeans + silhouette score (k in 3..6).
4. Name segments from their RFM profile (not just spend).
5. Train a classifier on the segments and report HELD-OUT accuracy, so the
   system can assign a segment to any new/updated customer reliably.
6. Save artefacts to backend/ml/ (the folder app.py and import_data.py read).
"""
import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, accuracy_score, f1_score,
    classification_report,
)

from preprocess import preprocess_data

RANDOM_STATE = 42
K_RANGE = range(3, 7)           # at least 3 segments so results are useful
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "customer_segmentation_50000_transactions.csv")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml")
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. RFM TABLE
df = preprocess_data(CSV_PATH)
snapshot_date = df["Date"].max() + pd.Timedelta(days=1)   # avoids Recency = 0

rfm = df.groupby("ACTIVE_CUSTOMER_ID").agg(
    Recency=("Date", lambda x: (snapshot_date - x.max()).days),
    Frequency=("Transaction ID", "count"),
    Monetary=("Total Amount", "sum"),
).reset_index().rename(columns={"ACTIVE_CUSTOMER_ID": "CustomerID"})

rfm["Avg_Order_Value"] = rfm["Monetary"] / rfm["Frequency"]

# 2. FEATURES + SCALING
# Log-transform removes skew; Avg_Order_Value is excluded because it is
# derived from Monetary/Frequency and only adds noise to the clusters.
FEATURES = ["Recency", "Frequency", "Monetary"]
X_log = np.log1p(rfm[FEATURES])
scaler = StandardScaler()
X = scaler.fit_transform(X_log)

# 3. CHOOSE K
print("\nFinding best K...")
best_k, best_score = None, -1
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)
    s = silhouette_score(X, km.labels_)
    print(f"K={k}  silhouette={s:.4f}")
    if s > best_score:
        best_k, best_score = k, s

# Compare against other algorithms at the same K (for the report only;
# KMeans is kept because it gives centroid-based, explainable segments).
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
comparison = {}
_km = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE).fit(X)
comparison["KMeans"] = silhouette_score(X, _km.labels_)
_gm = GaussianMixture(n_components=best_k, random_state=RANDOM_STATE).fit(X)
comparison["GaussianMixture"] = silhouette_score(X, _gm.predict(X))
_ag = AgglomerativeClustering(n_clusters=best_k).fit(X)
comparison["Agglomerative"] = silhouette_score(X, _ag.labels_)
print("\nAlgorithm comparison (silhouette):")
for name, val in comparison.items():
    print(f"  {name:16s} {val:.4f}")

kmeans = KMeans(n_clusters=best_k, n_init=50, random_state=RANDOM_STATE).fit(X)
rfm["Cluster"] = kmeans.labels_
best_score = silhouette_score(X, kmeans.labels_)
db_index = davies_bouldin_score(X, kmeans.labels_)

# 4. SEGMENT NAMES (from RFM profile)
centers = pd.DataFrame(kmeans.cluster_centers_, columns=FEATURES)
# higher = better customer: recent (low R), frequent, high spend
centers["Score"] = -centers["Recency"] + centers["Frequency"] + centers["Monetary"]
ranked = centers["Score"].sort_values(ascending=False).index.tolist()

NAME_SETS = {
    3: ["Champions", "Potential Loyalists", "At Risk Customers"],
    4: ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk Customers"],
    5: ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk Customers", "Lost Customers"],
    6: ["Champions", "Loyal Customers", "Potential Loyalists", "Regular Customers",
        "At Risk Customers", "Lost Customers"],
}
segment_names = {c: NAME_SETS[best_k][i] for i, c in enumerate(ranked)}
rfm["Segment"] = rfm["Cluster"].map(segment_names)

# 5. SEGMENT CLASSIFIER + HELD-OUT ACCURACY
y = rfm["Cluster"].values
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
clf.fit(X_tr, y_tr)
pred = clf.predict(X_te)
test_acc = accuracy_score(y_te, pred)
test_f1 = f1_score(y_te, pred, average="macro")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=cv)

print("\nHeld-out classification report")
print(classification_report(y_te, pred, target_names=[segment_names[i] for i in sorted(segment_names)]))

# final classifier uses all data
clf.fit(X, y)

# 6. SAVE
joblib.dump(kmeans, os.path.join(MODEL_DIR, "kmeans_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(clf, os.path.join(MODEL_DIR, "segment_classifier.pkl"))

# Frequency / Monetary stored as log values, same as the previous version
rfm_out = rfm.copy()
rfm_out["Frequency"] = np.log1p(rfm_out["Frequency"])
rfm_out["Monetary"] = np.log1p(rfm_out["Monetary"])
rfm_out.to_csv(os.path.join(MODEL_DIR, "customer_segments.csv"), index=False)

# Export a dependency-free copy of the model so the API can predict new
# customers without pandas / scikit-learn (useful on locked-down PCs).
model_export = {
    "features": FEATURES,
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "centroids": kmeans.cluster_centers_.tolist(),
    "segment_names": {str(c): n for c, n in segment_names.items()},
    "snapshot_date": str(snapshot_date.date()),
}
with open(os.path.join(MODEL_DIR, "model.json"), "w") as f:
    json.dump(model_export, f, indent=2)

metrics = {
    "customers": int(len(rfm)),
    "best_k": int(best_k),
    "silhouette_score": float(best_score),
    "davies_bouldin_index": float(db_index),
    "segment_classifier_accuracy": float(test_acc),
    "segment_classifier_cv_accuracy": float(cv_scores.mean()),
    "segment_classifier_macro_f1": float(test_f1),
    "algorithm_comparison_silhouette": {k: float(v) for k, v in comparison.items()},
    "segment_sizes": {segment_names[c]: int((rfm["Cluster"] == c).sum()) for c in segment_names},
}
with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("\nTraining completed")
print(json.dumps(metrics, indent=4))
