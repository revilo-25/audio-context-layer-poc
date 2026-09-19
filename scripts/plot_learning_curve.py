"""
plot_learning_curve.py
The classifier is a RandomForest (no gradient-descent loss curve exists).
As the closest analog requested by "loss curves (if trained)", this script
plots (a) out-of-bag error vs. number of trees, and (b) test accuracy vs.
training-set size (a learning curve), both standard diagnostics for
ensemble models.
"""
import json, os, numpy as np, librosa, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, "/home/claude/audio_qa_poc/scripts")
from train_event_detector import build_window_dataset, DATA_DIR

with open(os.path.join(DATA_DIR, "metadata.json")) as f:
    metadata = json.load(f)
with open(os.path.join(DATA_DIR, "clip_splits.json")) as f:
    splits = json.load(f)
metadata_by_id = {m["clip_id"]: m for m in metadata}

print("Building feature sets (cached from training run pattern)...")
X_train, y_train = build_window_dataset(splits["train"], metadata_by_id)
X_test, y_test = build_window_dataset(splits["test"], metadata_by_id)
scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

# (a) OOB error vs n_estimators
n_trees_list = [10, 25, 50, 100, 150, 200, 300, 400]
oob_errors = []
for nt in n_trees_list:
    clf = RandomForestClassifier(n_estimators=nt, oob_score=True, min_samples_leaf=2,
                                  class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    clf.fit(X_train_s, y_train)
    oob_errors.append(1 - clf.oob_score_)
    print(f"n_estimators={nt} OOB error={1-clf.oob_score_:.4f}")

# (b) learning curve: test accuracy vs training fraction
fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
train_accs, test_accs = [], []
rng = np.random.RandomState(0)
idx = rng.permutation(len(X_train_s))
for frac in fractions:
    n = max(50, int(frac * len(idx)))
    sub_idx = idx[:n]
    clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                  class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    clf.fit(X_train_s[sub_idx], y_train[sub_idx])
    train_accs.append(clf.score(X_train_s[sub_idx], y_train[sub_idx]))
    test_accs.append(clf.score(X_test_s, y_test))
    print(f"train_frac={frac} train_acc={train_accs[-1]:.4f} test_acc={test_accs[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(n_trees_list, oob_errors, marker="o")
axes[0].set_xlabel("Number of trees")
axes[0].set_ylabel("OOB error rate")
axes[0].set_title("Out-of-Bag Error vs. Ensemble Size\n(RandomForest convergence, analog of a loss curve)")
axes[0].grid(alpha=0.3)

axes[1].plot([f*100 for f in fractions], train_accs, marker="o", label="train accuracy")
axes[1].plot([f*100 for f in fractions], test_accs, marker="s", label="test accuracy")
axes[1].set_xlabel("% of training windows used")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Learning Curve: Accuracy vs. Training Set Size")
axes[1].legend()
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig("/home/claude/audio_qa_poc/figures/learning_curves.png", dpi=130)
print("Saved learning_curves.png")
