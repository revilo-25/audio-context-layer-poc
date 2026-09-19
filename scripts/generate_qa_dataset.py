"""
generate_qa_dataset.py
Builds question-answer pairs grounded in each clip's known event timeline.

Question types:
  what       - perceptual: which sounds are present / what environment
  counting   - how many times a given event class occurs
  temporal   - ordering: what happened before/after X, or which came first
  causal     - why a sound/environment appears the way it does

Answers are generated programmatically from ground truth, so every QA
pair is exactly correct by construction. This also gives us a clean
oracle to evaluate the PREDICTED pipeline (built on detected, not ground
truth, events) against.

Output: data/qa_pairs.json, data/{train,val,test}.json
"""
import json
import random
import os
from collections import Counter

random.seed(7)

DATA_DIR = "/home/claude/audio_qa_poc/data"
EVENT_NAMES_READABLE = {
    "dog_bark": "a dog barking", "car_horn": "a car horn", "footsteps": "footsteps",
    "rain": "rain", "phone_ring": "a phone ringing", "door_knock": "a door knock",
    "siren": "a siren", "glass_break": "glass breaking", "baby_cry": "a baby crying",
    "keyboard_typing": "keyboard typing",
}
ALL_EVENTS = list(EVENT_NAMES_READABLE.keys())

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

def build_qa_for_clip(meta):
    qs = []
    timeline = meta["timeline"]
    events_in_clip = [t["event"] for t in timeline]
    counts = Counter(events_in_clip)
    unique_events = sorted(set(events_in_clip))
    clip_id = meta["clip_id"]

    # ---------- WHAT (perceptual) ----------
    qs.append({
        "clip_id": clip_id, "type": "what",
        "question": "What sounds are present in this audio clip?",
        "answer": ", ".join(sorted(EVENT_NAMES_READABLE[e] for e in unique_events)),
    })
    qs.append({
        "clip_id": clip_id, "type": "what",
        "question": "What environment or scene does this audio most likely suggest?",
        "answer": meta["environment"],
    })
    # a present/absent check against a distractor event
    present_ev = random.choice(unique_events)
    absent_candidates = [e for e in ALL_EVENTS if e not in unique_events]
    if absent_candidates:
        absent_ev = random.choice(absent_candidates)
        for ev, expected in [(present_ev, "yes"), (absent_ev, "no")]:
            qs.append({
                "clip_id": clip_id, "type": "what",
                "question": f"Is there {EVENT_NAMES_READABLE[ev]} in this clip?",
                "answer": expected,
            })

    # ---------- COUNTING ----------
    for ev in unique_events:
        qs.append({
            "clip_id": clip_id, "type": "counting",
            "question": f"How many times does {EVENT_NAMES_READABLE[ev]} occur in this clip?",
            "answer": str(counts[ev]),
        })
    # a counting question about an absent event (answer 0) -- tests false positives
    if absent_candidates:
        ev0 = random.choice(absent_candidates)
        qs.append({
            "clip_id": clip_id, "type": "counting",
            "question": f"How many times does {EVENT_NAMES_READABLE[ev0]} occur in this clip?",
            "answer": "0",
        })

    # ---------- TEMPORAL ----------
    if len(timeline) >= 2:
        # first / last
        qs.append({
            "clip_id": clip_id, "type": "temporal",
            "question": "Which sound event occurs first in this clip?",
            "answer": EVENT_NAMES_READABLE[timeline[0]["event"]],
        })
        qs.append({
            "clip_id": clip_id, "type": "temporal",
            "question": "Which sound event occurs last in this clip?",
            "answer": EVENT_NAMES_READABLE[timeline[-1]["event"]],
        })
        # pairwise before/after, chosen from two distinct events
        distinct_pairs = [(a, b) for i, a in enumerate(timeline) for b in timeline[i + 1:]
                           if a["event"] != b["event"]]
        if distinct_pairs:
            a, b = random.choice(distinct_pairs)
            qs.append({
                "clip_id": clip_id, "type": "temporal",
                "question": f"Does {EVENT_NAMES_READABLE[a['event']]} occur before or after "
                             f"{EVENT_NAMES_READABLE[b['event']]}?",
                "answer": "before",
            })
            # what happens right after event a
            qs.append({
                "clip_id": clip_id, "type": "temporal",
                "question": f"What happens immediately after {EVENT_NAMES_READABLE[a['event']]}?",
                "answer": EVENT_NAMES_READABLE[b['event']],
            })

    # ---------- CAUSAL ----------
    for ev in unique_events[: min(2, len(unique_events))]:
        qs.append({
            "clip_id": clip_id, "type": "causal",
            "question": f"Why might {EVENT_NAMES_READABLE[ev]} be occurring in this recording?",
            "answer": CAUSES[ev],
        })
    qs.append({
        "clip_id": clip_id, "type": "causal",
        "question": "Why does this audio suggest that particular environment?",
        "answer": f"because it contains {', '.join(EVENT_NAMES_READABLE[e] for e in unique_events)}, "
                   f"a combination typically associated with a {meta['environment']}",
    })

    for q in qs:
        q["id"] = f"{clip_id}_{len(str(hash(q['question']))) }_{qs.index(q)}"
    return qs

def main():
    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        metadata = json.load(f)

    all_qa = []
    for meta in metadata:
        all_qa.extend(build_qa_for_clip(meta))

    # re-assign clean sequential ids
    for i, q in enumerate(all_qa):
        q["id"] = f"qa_{i:05d}"

    with open(os.path.join(DATA_DIR, "qa_pairs.json"), "w") as f:
        json.dump(all_qa, f, indent=2)

    # Split by CLIP (not by QA pair) to avoid leakage: 70/15/15
    clip_ids = [m["clip_id"] for m in metadata]
    random.shuffle(clip_ids)
    n = len(clip_ids)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train_ids = set(clip_ids[:n_train])
    val_ids = set(clip_ids[n_train:n_train + n_val])
    test_ids = set(clip_ids[n_train + n_val:])

    splits = {"train": [], "val": [], "test": []}
    for q in all_qa:
        if q["clip_id"] in train_ids:
            splits["train"].append(q)
        elif q["clip_id"] in val_ids:
            splits["val"].append(q)
        else:
            splits["test"].append(q)

    for name, rows in splits.items():
        with open(os.path.join(DATA_DIR, f"{name}.json"), "w") as f:
            json.dump(rows, f, indent=2)

    meta_by_split = {
        "train": sorted(train_ids), "val": sorted(val_ids), "test": sorted(test_ids)
    }
    with open(os.path.join(DATA_DIR, "clip_splits.json"), "w") as f:
        json.dump(meta_by_split, f, indent=2)

    print(f"Total QA pairs: {len(all_qa)}")
    print(f"Clips -> train/val/test: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")
    print(f"QA pairs -> train/val/test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
    type_counts = Counter(q["type"] for q in all_qa)
    print("QA type distribution:", dict(type_counts))

if __name__ == "__main__":
    main()
