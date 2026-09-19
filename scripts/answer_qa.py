"""
answer_qa.py
The "Audio Context Layer" inference pipeline:

  raw audio  --[event detector]-->  predicted event timeline
             --[question parser]-->  question type + slots
             --[answer engine]-->    natural-language answer

This module NEVER looks at ground-truth timelines when answering -- it
only uses what train_event_detector.py detected from the waveform, so
evaluation against the QA ground truth is a genuine end-to-end test of
the system, not an oracle lookup.
"""
import re
import json
import os
import numpy as np
import librosa
import joblib
from collections import Counter

DATA_DIR = "/home/claude/audio_qa_poc/data"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
MODEL_DIR = "/home/claude/audio_qa_poc/models"
SR = 16000
WIN, HOP = 0.5, 0.25

EVENT_NAMES_READABLE = {
    "dog_bark": "a dog barking", "car_horn": "a car horn", "footsteps": "footsteps",
    "rain": "rain", "phone_ring": "a phone ringing", "door_knock": "a door knock",
    "siren": "a siren", "glass_break": "glass breaking", "baby_cry": "a baby crying",
    "keyboard_typing": "keyboard typing",
}
READABLE_TO_EVENT = {v: k for k, v in EVENT_NAMES_READABLE.items()}

ENVIRONMENT_RULES = [
    ({"dog_bark", "footsteps", "car_horn"}, "street / outdoor urban scene"),
    ({"rain", "footsteps"}, "outdoors in the rain"),
    ({"phone_ring", "keyboard_typing"}, "office / indoor workspace"),
    ({"baby_cry", "door_knock"}, "home / domestic indoor scene"),
    ({"siren", "car_horn"}, "street with emergency vehicle traffic"),
    ({"glass_break", "dog_bark"}, "home intrusion / disturbance scene"),
]
CAUSES = {
    "dog_bark": "a dog reacting to a nearby stimulus (person, animal, or noise)",
    "car_horn": "a driver signaling another vehicle or pedestrian",
    "footsteps": "a person walking through or near the scene",
    "rain": "precipitation falling in the recorded environment",
    "phone_ring": "an incoming call on a nearby phone",
    "door_knock": "someone requesting entry at a door",
    "siren": "an emergency vehicle passing through the area",
    "glass_break": "an object breaking a glass surface, e.g. a window",
    "baby_cry": "an infant expressing distress or need",
    "keyboard_typing": "someone typing on a computer keyboard",
}

def infer_environment(event_set):
    best, best_score = "ambiguous / mixed scene", 0
    for rule_set, label in ENVIRONMENT_RULES:
        overlap = len(rule_set & event_set)
        if overlap > best_score:
            best_score = overlap
            best = label
    return best

# ---------------- Event detection (Audio Context Layer) ----------------

def _extract_window_features(y, sr):
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

class AudioContextLayer:
    def __init__(self, model_path=os.path.join(MODEL_DIR, "event_detector.joblib")):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.scaler = bundle["scaler"]

    def detect_timeline(self, audio_path, min_run=2, prob_threshold=0.35):
        """Slide a window over the clip, classify each hop, then merge
        consecutive same-label hops (with a minimum run length to
        suppress single-frame noise) into a predicted event timeline."""
        audio, sr = librosa.load(audio_path, sr=SR)
        n = len(audio)
        win_samps, hop_samps = int(WIN * SR), int(HOP * SR)
        feats, times = [], []
        for start in range(0, n - win_samps, hop_samps):
            seg = audio[start:start + win_samps]
            feats.append(_extract_window_features(seg, sr))
            times.append((start / SR, (start + win_samps) / SR))
        if not feats:
            return []
        X = self.scaler.transform(np.array(feats))
        probs = self.model.predict_proba(X)
        classes = self.model.classes_
        preds = []
        for p in probs:
            top_idx = np.argmax(p)
            label = classes[top_idx] if p[top_idx] >= prob_threshold else "background"
            preds.append(label)

        # merge consecutive identical labels (excluding background) into events
        timeline = []
        i = 0
        while i < len(preds):
            if preds[i] == "background":
                i += 1
                continue
            j = i
            while j + 1 < len(preds) and preds[j + 1] == preds[i]:
                j += 1
            run_len = j - i + 1
            if run_len >= min_run or (j == len(preds) - 1 and run_len >= 1):
                timeline.append({
                    "event": preds[i],
                    "start": round(times[i][0], 2),
                    "end": round(times[j][1], 2),
                })
            i = j + 1
        # merge adjacent same-event windows that ended up split by a stray gap
        merged = []
        for ev in timeline:
            if merged and merged[-1]["event"] == ev["event"] and ev["start"] - merged[-1]["end"] < WIN:
                merged[-1]["end"] = ev["end"]
            else:
                merged.append(dict(ev))
        return merged

# ---------------- Question parsing + answer engine ----------------

def _find_event_in_question(question):
    q = question.lower()
    matches = []
    for readable, ev in READABLE_TO_EVENT.items():
        r = readable.lower().replace("a ", "").replace("an ", "")
        if r in q:
            matches.append(ev)
    return matches

def answer_question(question, qtype, timeline):
    """timeline: list of {"event","start","end"} dicts, PREDICTED (or ground truth
    when evaluating the oracle upper bound)."""
    events_sorted = sorted(timeline, key=lambda x: x["start"])
    unique_events = sorted(set(e["event"] for e in events_sorted))
    counts = Counter(e["event"] for e in events_sorted)
    event_set = set(unique_events)

    if qtype == "what":
        if "environment" in question.lower() or "scene" in question.lower():
            return infer_environment(event_set)
        if question.lower().startswith("is there"):
            mentioned = _find_event_in_question(question)
            if mentioned:
                return "yes" if mentioned[0] in event_set else "no"
            return "no"
        # "what sounds are present"
        return ", ".join(EVENT_NAMES_READABLE[e] for e in unique_events) if unique_events else "no clear sounds detected"

    if qtype == "counting":
        mentioned = _find_event_in_question(question)
        if mentioned:
            return str(counts.get(mentioned[0], 0))
        return "0"

    if qtype == "temporal":
        ql = question.lower()
        if not events_sorted:
            return "no events detected"
        if "first" in ql:
            return EVENT_NAMES_READABLE[events_sorted[0]["event"]]
        if "last" in ql:
            return EVENT_NAMES_READABLE[events_sorted[-1]["event"]]
        if "before or after" in ql:
            mentioned = _find_event_in_question(question)
            if len(mentioned) >= 2:
                a, b = mentioned[0], mentioned[1]
                a_first = next((e for e in events_sorted if e["event"] == a), None)
                b_first = next((e for e in events_sorted if e["event"] == b), None)
                if a_first and b_first:
                    return "before" if a_first["start"] < b_first["start"] else "after"
            return "before"
        if "immediately after" in ql:
            mentioned = _find_event_in_question(question)
            if mentioned:
                a = mentioned[0]
                a_ev = next((e for e in events_sorted if e["event"] == a), None)
                if a_ev:
                    later = [e for e in events_sorted if e["start"] > a_ev["start"]]
                    if later:
                        return EVENT_NAMES_READABLE[later[0]["event"]]
            return "nothing else detected"
        return "unknown"

    if qtype == "causal":
        ql = question.lower()
        if "environment" in ql:
            env = infer_environment(event_set)
            return (f"because it contains {', '.join(EVENT_NAMES_READABLE[e] for e in unique_events)}, "
                    f"a combination typically associated with a {env}")
        mentioned = _find_event_in_question(question)
        if mentioned:
            return CAUSES.get(mentioned[0], "an unidentified environmental cause")
        return "an unidentified environmental cause"

    return "unsupported question type"

# ---------------- Simple heuristic question-type classifier ----------------

def classify_question_type(question):
    q = question.lower()
    if q.startswith("how many"):
        return "counting"
    if any(k in q for k in ["before", "after", "first", "last", "immediately"]):
        return "temporal"
    if q.startswith("why"):
        return "causal"
    return "what"

if __name__ == "__main__":
    layer = AudioContextLayer()
    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    sample = metadata[0]
    tl = layer.detect_timeline(os.path.join(AUDIO_DIR, sample["file"]))
    print("Ground truth timeline:", sample["timeline"])
    print("Predicted timeline:", tl)
    for q in ["What sounds are present in this audio clip?",
              "How many times does keyboard typing occur in this clip?",
              "Which sound event occurs first in this clip?"]:
        print(q, "->", answer_question(q, classify_question_type(q), tl))
