"""
train_event_detector.py
Trains the "Audio Context" understanding layer: a sliding-window multi-class
classifier over MFCC + spectral features that predicts, for each 0.5s hop,
which event class (or 'background') is active. Aggregating hop-level
predictions over a clip gives us a predicted event timeline, which is what
the QA answering pipeline (answer_qa.py) consumes -- i.e. the model never
sees ground-truth timelines at inference/answering time, only what it
detected.

This is intentionally a lightweight, CPU-only, reproducible model
(RandomForest over hand-crafted features) rather than a deep spectrogram
model, appropriate for a small synthetic PoC dataset (see docs for
justification).
"""
import json
import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATA_DIR = "/home/claude/audio_qa_poc/data"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
MODEL_DIR = "/home/claude/audio_qa_poc/models"
RESULTS_DIR = "/home/claude/audio_qa_poc/results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

SR = 16000
WIN = 0.5      # window length (s) for sliding-window labeling
HOP = 0.25     # hop length (s)

def extract_window_features(y, sr):
    """Hand-crafted feature vector for one window: MFCCs + spectral stats + ZCR + RMS."""
    if len(y) < 512:
        y = np.pad(y, (0, 512 - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean, mfcc_std = mfcc.mean(axis=1), mfcc.std(axis=1)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zcr = librosa.feature.zero_crossing_rate(y=y).mean()
    rms = librosa.feature.rms(y=y).mean()
    flatness = librosa.feature.spectral_flatness(y=y).mean()
    return np.concatenate([mfcc_mean, mfcc_std, [centroid, bandwidth, rolloff, zcr, rms, flatness]])

def label_for_window(t_start, t_end, timeline):
    """Assign the event with greatest temporal overlap in this window, else 'background'."""
    best_ev, best_overlap = "background", 0.0
    for ev in timeline:
        overlap = max(0.0, min(t_end, ev["end"]) - max(t_start, ev["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_ev = ev["event"]
    return best_ev

def build_window_dataset(clip_ids, metadata_by_id):
    X, y = [], []
    for cid in clip_ids:
        meta = metadata_by_id[cid]
        path = os.path.join(AUDIO_DIR, meta["file"])
        audio, sr = librosa.load(path, sr=SR)
        n = len(audio)
        win_samps, hop_samps = int(WIN * SR), int(HOP * SR)
        for start in range(0, n - win_samps, hop_samps):
            seg = audio[start:start + win_samps]
            t0, t1 = start / SR, (start + win_samps) / SR
            label = label_for_window(t0, t1, meta["timeline"])
            X.append(extract_window_features(seg, sr))
            y.append(label)
    return np.array(X), np.array(y)

def main():
    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    with open(os.path.join(DATA_DIR, "clip_splits.json")) as f:
        splits = json.load(f)
    metadata_by_id = {m["clip_id"]: m for m in metadata}

    print("Extracting features for train split...")
    X_train, y_train = build_window_dataset(splits["train"], metadata_by_id)
    print("Extracting features for val split...")
    X_val, y_val = build_window_dataset(splits["val"], metadata_by_id)
    print("Extracting features for test split...")
    X_test, y_test = build_window_dataset(splits["test"], metadata_by_id)

    print(f"Window counts -> train {len(y_train)}, val {len(y_val)}, test {len(y_test)}")

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    )
    clf.fit(X_train_s, y_train)

    val_acc = clf.score(X_val_s, y_val)
    test_acc = clf.score(X_test_s, y_test)
    print(f"Window-level val accuracy: {val_acc:.4f}")
    print(f"Window-level test accuracy: {test_acc:.4f}")

    y_pred = clf.predict(X_test_s)
    report = classification_report(y_test, y_pred, zero_division=0)
    labels_sorted = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    with open(os.path.join(RESULTS_DIR, "event_detector_report.txt"), "w") as f:
        f.write(f"Window-level validation accuracy: {val_acc:.4f}\n")
        f.write(f"Window-level test accuracy: {test_acc:.4f}\n\n")
        f.write("Classification report (test set):\n")
        f.write(report)
        f.write("\n\nConfusion matrix labels: " + ", ".join(labels_sorted) + "\n")
        f.write(np.array2string(cm))

    joblib.dump({"model": clf, "scaler": scaler, "labels": sorted(clf.classes_)},
                os.path.join(MODEL_DIR, "event_detector.joblib"))

    # save confusion matrix figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels_sorted)))
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_xticklabels(labels_sorted, rotation=90)
    ax.set_yticklabels(labels_sorted)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Event Detector Confusion Matrix (window-level, test set)")
    for i in range(len(labels_sorted)):
        for j in range(len(labels_sorted)):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig("/home/claude/audio_qa_poc/figures/confusion_matrix.png", dpi=130)
    print("Saved confusion matrix figure.")

    # feature importance plot (top 15)
    importances = clf.feature_importances_
    feat_names = ([f"mfcc_mean_{i}" for i in range(13)] + [f"mfcc_std_{i}" for i in range(13)] +
                  ["centroid", "bandwidth", "rolloff", "zcr", "rms", "flatness"])
    idx = np.argsort(importances)[::-1][:15]
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.barh([feat_names[i] for i in idx][::-1], importances[idx][::-1])
    ax2.set_title("Top-15 Feature Importances (RandomForest)")
    fig2.tight_layout()
    fig2.savefig("/home/claude/audio_qa_poc/figures/feature_importance.png", dpi=130)
    print("Saved feature importance figure.")

if __name__ == "__main__":
    main()
