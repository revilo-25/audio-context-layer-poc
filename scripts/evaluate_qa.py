"""
evaluate_qa.py
End-to-end evaluation of the Audio Context Layer on the held-out TEST split.
For every clip in the test set:
  1. run the trained event detector to get a PREDICTED timeline (no oracle access)
  2. answer every QA pair associated with that clip using only the predicted timeline
  3. score the predicted answer against the ground-truth answer

Metrics:
  - Exact-match accuracy, overall and broken down by question type
  - Numeric tolerance accuracy for counting (exact, since counts are small ints)
  - Set/keyword overlap (F1) for free-text answers (what-sounds-present, causal)
  - Confusion analysis: which clips/questions fail and why
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from answer_qa import AudioContextLayer, answer_question, classify_question_type

DATA_DIR = "/home/claude/audio_qa_poc/data"
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
RESULTS_DIR = "/home/claude/audio_qa_poc/results"

def normalize(s):
    return s.strip().lower().replace(".", "")

def token_set(s):
    return set(re.sub(r"[^a-z0-9 ]", "", s.lower()).split())

import re

def score_answer(pred, gold, qtype):
    p, g = normalize(pred), normalize(gold)
    if qtype in ("counting",):
        return 1.0 if p == g else 0.0
    if qtype == "temporal" and g in ("before", "after", "yes", "no"):
        return 1.0 if p == g else 0.0
    if qtype == "what" and g in ("yes", "no"):
        return 1.0 if p == g else 0.0
    if p == g:
        return 1.0
    # token-level F1 for free-text answers (event lists, environment, causal reasoning)
    pt, gt = token_set(pred), token_set(gold)
    if not pt or not gt:
        return 0.0
    inter = len(pt & gt)
    prec = inter / len(pt)
    rec = inter / len(gt)
    if prec + rec == 0:
        return 0.0
    f1 = 2 * prec * rec / (prec + rec)
    return f1

def main():
    with open(os.path.join(DATA_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    metadata_by_id = {m["clip_id"]: m for m in metadata}
    with open(os.path.join(DATA_DIR, "test.json")) as f:
        test_qa = json.load(f)
    with open(os.path.join(DATA_DIR, "clip_splits.json")) as f:
        test_clip_ids = json.load(f)["test"]

    layer = AudioContextLayer()

    print("Running event detector on all test clips...")
    predicted_timelines = {}
    for cid in test_clip_ids:
        meta = metadata_by_id[cid]
        path = os.path.join(AUDIO_DIR, meta["file"])
        predicted_timelines[cid] = layer.detect_timeline(path)

    records = []
    for qa in test_qa:
        cid = qa["clip_id"]
        tl = predicted_timelines[cid]
        pred_type = classify_question_type(qa["question"])
        pred_answer = answer_question(qa["question"], pred_type, tl)
        score = score_answer(pred_answer, qa["answer"], qa["type"])
        records.append({
            "id": qa["id"], "clip_id": cid, "type": qa["type"],
            "predicted_type": pred_type,
            "question": qa["question"], "gold_answer": qa["answer"],
            "pred_answer": pred_answer, "score": score,
        })

    # ---------- aggregate metrics ----------
    from collections import defaultdict
    by_type_scores = defaultdict(list)
    for r in records:
        by_type_scores[r["type"]].append(r["score"])
    overall = sum(r["score"] for r in records) / len(records)

    type_qclf_correct = sum(1 for r in records if r["predicted_type"] == r["type"])
    type_qclf_acc = type_qclf_correct / len(records)

    lines = []
    lines.append(f"N test QA pairs: {len(records)}")
    lines.append(f"Question-type classification accuracy: {type_qclf_acc:.4f}")
    lines.append(f"Overall answer score (exact-match / F1 hybrid): {overall:.4f}\n")
    lines.append("Per-question-type breakdown:")
    for qtype in ["what", "counting", "temporal", "causal"]:
        scores = by_type_scores[qtype]
        avg = sum(scores) / len(scores) if scores else float("nan")
        exact = sum(1 for s in scores if s == 1.0) / len(scores) if scores else float("nan")
        lines.append(f"  {qtype:10s} n={len(scores):4d}  avg_score={avg:.4f}  exact_match_rate={exact:.4f}")

    report_text = "\n".join(lines)
    print(report_text)

    with open(os.path.join(RESULTS_DIR, "qa_eval_report.txt"), "w") as f:
        f.write(report_text + "\n")

    with open(os.path.join(RESULTS_DIR, "qa_eval_records.json"), "w") as f:
        json.dump(records, f, indent=2)

    # ---------- error analysis: worst-scoring examples per type ----------
    err_lines = ["ERROR ANALYSIS: lowest-scoring examples per question type\n"]
    for qtype in ["what", "counting", "temporal", "causal"]:
        subset = sorted([r for r in records if r["type"] == qtype], key=lambda r: r["score"])[:6]
        err_lines.append(f"\n--- {qtype.upper()} ---")
        for r in subset:
            err_lines.append(f"  clip={r['clip_id']} score={r['score']:.2f}")
            err_lines.append(f"    Q: {r['question']}")
            err_lines.append(f"    gold: {r['gold_answer']!r}")
            err_lines.append(f"    pred: {r['pred_answer']!r}")

    err_text = "\n".join(err_lines)
    with open(os.path.join(RESULTS_DIR, "error_analysis.txt"), "w") as f:
        f.write(err_text + "\n")
    print("\nSaved error analysis to results/error_analysis.txt")

if __name__ == "__main__":
    main()
