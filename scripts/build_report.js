const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, ImageRun, PageBreak,
  LevelFormat, convertInchesToTwip,
} = require("docx");

const FIG = "/home/claude/audio_qa_poc/figures";

function H1(text) { return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 320, after: 160 } }); }
function H2(text) { return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 240, after: 120 } }); }
function H3(text) { return new Paragraph({ text, heading: HeadingLevel.HEADING_3, spacing: { before: 180, after: 100 } }); }
function P(text, opts = {}) {
  return new Paragraph({ spacing: { after: 160 }, children: [new TextRun({ text, ...opts })] });
}
function PB(runs) { return new Paragraph({ spacing: { after: 160 }, children: runs }); }
function bullet(text, level = 0) {
  return new Paragraph({ text, bullet: { level }, spacing: { after: 80 } });
}
function codeP(text) {
  return new Paragraph({
    spacing: { after: 80 },
    children: [new TextRun({ text, font: "Consolas", size: 18 })],
  });
}
function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "1F4E79" } : undefined,
    children: [new Paragraph({
      children: [new TextRun({ text, bold: !!opts.header, color: opts.header ? "FFFFFF" : "000000", size: 20 })],
    })],
  });
}
function table(headers, rows, widths) {
  const w = widths || headers.map(() => Math.floor(9000 / headers.length));
  return new Table({
    width: { size: 9000, type: WidthType.DXA },
    columnWidths: w,
    rows: [
      new TableRow({ children: headers.map((h, i) => cell(h, { header: true, width: w[i] })) }),
      ...rows.map(r => new TableRow({ children: r.map((c, i) => cell(String(c), { width: w[i] })) })),
    ],
  });
}
function image(path, width, height) {
  const data = fs.readFileSync(path);
  return new Paragraph({
    spacing: { after: 200 },
    alignment: AlignmentType.CENTER,
    children: [new ImageRun({ type: "png", data, transformation: { width, height } })],
  });
}
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [new TextRun({ text, italics: true, size: 18 })],
  });
}

const doc = new Document({
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 } } },
    children: [
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 80 },
        children: [new TextRun({ text: "Audio Context Layer", bold: true, size: 44 })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 40 },
        children: [new TextRun({ text: "A Proof-of-Concept System for Audio Question Answering", size: 26, italics: true })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 400 },
        children: [new TextRun({ text: "Technical Report", size: 22, color: "555555" })],
      }),

      H1("1. Problem Formulation"),
      P("The task is to build a system that, given an audio clip, can build a contextual understanding of the sounds it contains and answer natural language questions grounded in that audio. This is the audio analogue of Visual Question Answering (VQA): instead of an image + question -> answer, we require audio + question -> answer."),
      P("We decompose \"understanding\" into four question families, each of which stresses a different capability of the system:"),
      bullet("Perceptual / \"what\" questions -- open-set recognition of sound sources and coarse scene/environment inference (e.g. \"What sounds are present?\", \"What environment does this suggest?\")."),
      bullet("Counting questions -- event detection must be discretized into countable instances, not just clip-level tags (e.g. \"How many times does a dog bark?\")."),
      bullet("Temporal questions -- the system must localize events in time and reason about their order (e.g. \"What happens after the door knock?\", \"Which occurs first?\")."),
      bullet("Causal / reasoning questions -- the system must connect a detected acoustic event to a plausible real-world explanation, going beyond raw classification into commonsense grounding (e.g. \"Why might a siren be occurring here?\")."),
      P("Formally, we define the system as a function f(audio, question) -> answer, and require that f be decomposable into an audio-understanding stage (which does not see the question) and a reasoning/answer stage (which is audio-agnostic given the extracted context). This decomposition is deliberate: it makes the \"Audio Context Layer\" a reusable, inspectable intermediate representation -- a predicted event timeline -- rather than an opaque end-to-end black box, which materially helps debugging, error analysis, and answer justification."),

      H1("2. Research Study"),
      P("Before designing the PoC we surveyed the existing landscape of audio question answering and adjacent tasks to ground our design decisions:"),
      H3("2.1 Audio Question Answering datasets and models"),
      bullet("Clotho-AQA (Lipping et al., 2022) -- ~1991 real-world audio clips with 5 crowd-sourced QA pairs each, mostly yes/no and single-word answers, evaluated with a binary-answer LSTM/attention baseline. Established that even simple binary-QA over natural audio is far from solved with small models."),
      bullet("DAQA (Fayek & Johnson, 2020) -- a synthetically generated diagnostic AQA dataset (analogous to CLEVR for vision) built by programmatically combining sound events with known timing, specifically to test counting, temporal ordering, and comparison reasoning in isolation from perception noise. This is the closest prior art to our approach and directly motivated our decision to synthesize a fully-annotated dataset rather than rely on noisy crowd-sourced labels for the reasoning-heavy question types."),
      bullet("MusicAVQA / AVQA -- extend AQA into the audio-visual setting; useful for question-taxonomy ideas (existential, counting, comparative, temporal) but out of scope since we are audio-only."),
      bullet("CLAP / LAION-CLAP and PANNs (Pretrained Audio Neural Networks) -- large pretrained audio-text and audio-tagging embedding models that give strong zero-shot perceptual grounding for \"what\" questions in production systems, but are heavy (100M+ params), need GPU, and would be difficult to inspect/debug for a PoC with an interpretable, from-scratch pipeline."),
      bullet("Cascaded / modular VQA architectures (e.g. classical VQA baselines that first extract objects then reason over them symbolically) -- informed our own \"detect events -> symbolic answer engine\" cascade, chosen over an end-to-end sequence model for its transparency and small-data sample efficiency."),
      H3("2.2 Key design implications drawn from the study"),
      bullet("Counting and temporal reasoning are the hardest question types across the literature (DAQA reports the largest performance gaps here) -- this motivated building our evaluation with explicit per-type breakdowns from the start, and predicting our own system would show the same weakness (confirmed in Section 6)."),
      bullet("Purely crowd-sourced AQA datasets (Clotho-AQA) cannot cleanly separate perceptual errors from reasoning errors, because ground truth itself is noisy. A synthetic, fully-controlled dataset (as in DAQA) lets us attribute every error to either the perception stage (event detector) or the reasoning stage (answer engine) -- this is the central methodological choice of this PoC."),
      bullet("Given CPU-only, no-internet-dataset-download constraints for this PoC, a from-scratch synthetic corpus plus a lightweight, classical ML event detector was judged the most reproducible and inspectable path to a genuinely working end-to-end system within the time budget, versus attempting to fine-tune a large pretrained audio-language model."),

      H1("3. Dataset"),
      H3("3.1 Why a synthetic, programmatically-labeled dataset"),
      P("We curate our own synthetic dataset rather than using an existing one, for three reasons that follow directly from the research study above: (1) exact, noise-free ground truth for every event's identity and timestamps is required to build objective counting/temporal/causal QA pairs and to separate perception errors from reasoning errors during evaluation; (2) the sandboxed environment has no access to third-party audio dataset hosts; (3) full control over the event vocabulary lets us guarantee balanced coverage of all four question types."),
      H3("3.2 Audio synthesis"),
      P("Ten sound-event classes were implemented as parametric waveform synthesizers in NumPy (16 kHz mono): dog_bark, car_horn, footsteps, rain, phone_ring, door_knock, siren, glass_break, baby_cry, keyboard_typing. Each synthesizer combines tonal, noise-burst, or amplitude-modulated components with an attack/release envelope, e.g. footsteps are periodic transient clicks, rain is filtered broadband noise, a siren is a frequency-modulated tone, dog_bark is a decaying square-wave burst with noise."),
      P("220 ten-second \"soundscape\" clips were generated. Each clip places 2-5 non-overlapping events (sampled without replacement from the 10-class vocabulary) at randomized, non-overlapping start times, plus a small amount of background hiss. Because placement is programmatic, we retain an exact ground-truth event timeline (event label, start time, end time) for every clip."),
      H3("3.3 Environment / scene labels and causal templates"),
      P("A small rule table maps event co-occurrence patterns to one of 6 coarse environment labels (e.g. {dog_bark, footsteps, car_horn} -> \"street / outdoor urban scene\"; {phone_ring, keyboard_typing} -> \"office / indoor workspace\"), with an \"ambiguous / mixed scene\" fallback. A second table maps each event class to a one-sentence plausible real-world cause (e.g. siren -> \"an emergency vehicle passing through the area\"), used to generate causal QA answers."),
      H3("3.4 Question-answer generation"),
      P("For every clip, QA pairs are generated programmatically across all four types using the exact ground-truth timeline:"),
      bullet("what: sound inventory, environment inference, and yes/no presence checks against both a present and an absent (distractor) event class."),
      bullet("counting: \"how many times does X occur\" for every event class present, plus one distractor class known to be absent (expected answer 0), to explicitly test for detector false positives."),
      bullet("temporal: first/last event, pairwise before/after between two distinct event instances, and \"what happens immediately after X\"."),
      bullet("causal: \"why might event X be occurring\" for up to 2 events per clip, and \"why does this audio suggest that environment\", using the cause/environment lookup tables."),
      P("This produced 3,386 QA pairs total. Distribution across types: what = 880, counting = 966, temporal = 880, causal = 660."),
      H3("3.5 Train / validation / test split"),
      P("The split is performed at the CLIP level (not the QA-pair level) to prevent leakage -- all QA pairs from a given clip stay in the same split. 220 clips were split 70% / 15% / 15%, giving 154 / 33 / 33 clips and 2,365 / 512 / 509 QA pairs for train / val / test respectively."),
      table(
        ["Split", "# Clips", "# QA pairs", "what", "counting", "temporal", "causal"],
        [
          ["Train", 154, 2365, "~615", "~676", "~615", "~459"],
          ["Val", 33, 512, "~133", "~146", "~133", "~100"],
          ["Test", 33, 509, "132", "146", "132", "99"],
        ],
        [1200, 1500, 1500, 1200, 1500, 1500, 1200]
      ),
      new Paragraph({ text: "", spacing: { after: 160 } }),
      H3("3.6 Data format"),
      P("Files are stored under data/: audio/clip_XXXX.wav (16 kHz mono WAV), metadata.json (per-clip event timeline + environment label), qa_pairs.json (all QA pairs with clip_id, type, question, answer, id), clip_splits.json (clip-id lists per split), and train.json / val.json / test.json (QA pairs pre-filtered by split)."),

      H1("4. Method"),
      H3("4.1 Architecture overview"),
      P("The system is a modular, cascaded pipeline rather than a single end-to-end neural model, by deliberate design choice (justified below):"),
      bullet("Audio Context Layer (perception): a sliding-window feature extractor + classifier converts a raw waveform into a predicted event timeline -- a list of (event_label, start_time, end_time) tuples -- without ever seeing the question."),
      bullet("Question parser: a lightweight rule-based classifier maps the natural-language question to one of the four question types and extracts \"slots\" (which event names are mentioned, comparison keywords such as before/after/first/last)."),
      bullet("Answer engine: a symbolic reasoning module that combines the predicted timeline with the parsed question to compute the final answer (counting via Counter, temporal via timestamp comparison, causal via lookup table, perceptual via set operations and the environment-inference rule table)."),
      H3("4.2 Why a cascaded, feature-based design instead of an end-to-end deep model"),
      bullet("Sample efficiency: with only 220 clips, a from-scratch deep audio model (e.g. a CNN over spectrograms or a transformer) would badly overfit; classical features (MFCCs + spectral statistics) plus a Random Forest generalize far better in this low-data regime and train in seconds on CPU."),
      bullet("Interpretability & debuggability: because the timeline is an explicit intermediate artifact, every wrong answer can be traced to either a perception failure (wrong/missed event in the timeline) or a reasoning failure (correct timeline, wrong logic) -- this directly enables the error analysis in Section 6."),
      bullet("Counting and temporal questions require instance-level, time-localized detection, not just clip-level tags -- so the perception stage is explicitly windowed (0.5s windows, 0.25s hop) and instance-detected via run-length merging, rather than a single whole-clip multi-label classifier."),
      bullet("Reproducibility: no GPU, no pretrained weights, and no external dataset downloads are required -- the entire pipeline runs deterministically from the two dataset-generation scripts through training to evaluation."),
      H3("4.3 Audio Context Layer (perception) details"),
      P("Each 10s clip is scanned with a 0.5s window and 0.25s hop (75% overlap) to give fine-grained temporal resolution. For every window we extract a 32-dimensional hand-crafted feature vector: 13 MFCC means + 13 MFCC standard deviations (timbral content), spectral centroid, bandwidth, roll-off, zero-crossing rate, RMS energy, and spectral flatness (captures noise-like vs. tonal events, important for distinguishing e.g. rain from glass_break)."),
      P("Each window is labeled during training with the event class of greatest temporal overlap in that window, or \"background\" if no event overlaps it (majority-overlap labeling). A Random Forest classifier (300 trees, class-balanced subsampling to counter the large \"background\" majority class, standardized features) is trained on these window-level (feature, label) pairs."),
      P("At inference time, windows are classified independently (via predict_proba with a 0.35 confidence threshold to suppress low-confidence background/foreground confusion), and consecutive windows sharing a predicted label are merged into event instances (run-length merging), which discretizes the frame-level output into the countable, temporally-bounded event instances that the answer engine requires."),
      H3("4.4 Answer engine details"),
      P("The answer engine is fully symbolic (no learned weights): it operates purely on the predicted timeline and the question's parsed type/slots. This was a deliberate simplification given dataset scale -- with a larger dataset a natural extension would be a learned reasoning module (e.g. a small sequence model over the timeline + question embedding); Section 7 discusses this as future work."),

      H1("5. Experimental Setup"),
      table(
        ["Component", "Configuration"],
        [
          ["Sample rate", "16 kHz mono"],
          ["Clip length", "10 s"],
          ["Sliding window / hop", "0.5 s / 0.25 s (75% overlap)"],
          ["Features", "13 MFCC mean + 13 MFCC std + centroid + bandwidth + rolloff + ZCR + RMS + flatness (32-d)"],
          ["Classifier", "RandomForestClassifier, n_estimators=300, min_samples_leaf=2, class_weight=balanced_subsample, random_state=42"],
          ["Detection threshold", "0.35 predicted probability; run-length >= 2 windows to register an event instance"],
          ["Train / Val / Test", "154 / 33 / 33 clips (clip-level split, seed=42/7)"],
          ["Eval metric (event detector)", "Window-level accuracy, precision/recall/F1 per class, confusion matrix"],
          ["Eval metric (QA)", "Exact-match for counting/binary/temporal-directional answers; token-F1 for free-text answers (sound lists, environment, causal); question-type classification accuracy"],
        ],
        [3000, 6000]
      ),
      new Paragraph({ text: "", spacing: { after: 160 } }),
      P("All randomness is seeded (NumPy/random seeds 42 and 7 for dataset generation, random_state=42 for the classifier) for full reproducibility. The entire pipeline -- dataset generation, QA generation, feature extraction, training, and evaluation -- runs via five scripts in sequence with no manual steps."),

      H1("6. Results"),
      H3("6.1 Event detector (perception stage)"),
      P("Window-level accuracy on the held-out test set: 95.37% (validation: 95.22%), with macro-averaged F1 of 0.94 across the 10 event classes + background."),
      image(`${FIG}/confusion_matrix.png`, 470, 430),
      caption("Figure 1. Window-level confusion matrix on the test set. The dominant confusion is glass_break being missed as background (short, transient event vs. 0.5s analysis window), and secondary confusion between footsteps/door_knock and background (both are also short percussive events)."),
      image(`${FIG}/feature_importance.png`, 430, 330),
      caption("Figure 2. Top-15 most important features. MFCC coefficients and spectral flatness dominate, consistent with these features capturing timbral / noise-vs-tonal distinctions between event classes."),
      H3("6.2 End-to-end QA evaluation (test set, 509 QA pairs)"),
      P("Question-type classification (the rule-based parser) reached 100% accuracy on the test set, meaning all failures downstream are perception or reasoning failures, not misrouted questions."),
      table(
        ["Question type", "N", "Avg. score", "Exact-match rate"],
        [
          ["what", 132, "0.989", "97.7%"],
          ["counting", 146, "0.986", "98.6%"],
          ["temporal", 132, "0.791", "78.0%"],
          ["causal", 99, "0.999", "98.0%"],
          ["Overall", 509, "0.939", "--"],
        ],
        [2200, 1500, 2200, 3100]
      ),
      new Paragraph({ text: "", spacing: { after: 160 } }),
      P("Overall end-to-end QA score: 93.9% (hybrid exact-match / token-F1 metric, Section 5). Causal and \"what\" questions score highest because they only require correct set-level detection (which events are present) and can tolerate a single mis-detected event via partial token-F1 credit. Counting is high because most clips have small (0-1) true counts per class and the detector rarely double-counts or drops a full instance. Temporal questions are markedly the hardest question type, consistent with the DAQA finding discussed in Section 2."),
      H3("6.3 Loss / convergence curves"),
      P("The event detector is a Random Forest, which has no gradient-descent loss curve; as the closest valid analogue we report (a) out-of-bag (OOB) error vs. number of trees, which is RandomForest's standard convergence diagnostic, and (b) a learning curve of train/test accuracy vs. training-set size."),
      image(`${FIG}/learning_curves.png`, 620, 240),
      caption("Figure 3. Left: OOB error stabilizes by ~150-200 trees (diminishing returns beyond), justifying the 300-tree configuration used. Right: test accuracy improves from 92.9% (10% of training data) to 95.1% (100%) with a shrinking marginal gain, and the train/test gap (~4-5 points) indicates mild but not severe overfitting -- consistent with using a fairly small, synthetic, low-diversity dataset."),

      H1("7. Observations and Limitations"),
      H3("7.1 Observations"),
      bullet("Errors are dominated by one acoustic property, not a single event class: short, low-energy transient events (glass_break, and to a lesser extent footsteps/door_knock) are systematically under-detected because a 0.5s analysis window with a 2-window minimum run-length is a poor match for events shorter than ~0.4s. This directly explains both the confusion-matrix pattern (Fig. 1) and the specific counting failures seen in error analysis (glass_break undercounted in 2 of its test occurrences)."),
      bullet("Temporal reasoning errors are almost entirely inherited from small timing perturbations in the predicted timeline (a car_horn detected as starting a few hundred ms later than ground truth can flip a close before/after pairwise ordering), not from the reasoning logic itself, which is exact given a correct timeline (verified by re-running the answer engine directly on ground-truth timelines, which scores 100% on all temporal questions)."),
      bullet("The modular design successfully localizes failures: every incorrect QA answer could be attributed to a specific upstream cause (a missed/extra event, a boundary-timing shift, or -- never observed in this run -- a reasoning bug), validating the cascaded architecture choice in Section 4.2."),
      H3("7.2 Limitations"),
      bullet("Synthetic-only data: all audio is programmatically synthesized, not real-world recordings. Absolute accuracy numbers will not transfer directly to natural audio, which has far more acoustic variability, overlapping/simultaneous events, and ambiguous scene semantics."),
      bullet("No overlapping events: by construction, events never overlap in time, which sidesteps a genuinely hard and common real-audio problem (polyphonic sound event detection)."),
      bullet("Small vocabulary and rule-based semantics: only 10 event classes and a hand-authored environment/causal lookup table; the causal \"reasoning\" is template retrieval, not generative commonsense inference, so it cannot answer causal questions outside the fixed template set."),
      bullet("Symbolic answer engine: because the reasoning stage has no learned parameters, it cannot handle free-form or paraphrased questions outside the templated question patterns without extending the rule-based parser."),
      bullet("Window/threshold sensitivity: detection quality is sensitive to the 0.5s window and 0.35 probability threshold, which were hand-tuned rather than formally optimized via a validation sweep; short events remain a known weak point."),
      H3("7.3 Suggested extensions"),
      bullet("Replace the window classifier with a small 1D-CNN or CRNN over log-mel spectrograms once more (ideally real) data is available, enabling variable-length receptive fields that better capture short transients like glass breaking."),
      bullet("Extend to overlapping/polyphonic event synthesis and add multi-label window classification (sigmoid outputs per class) instead of single-label softmax."),
      bullet("Replace the symbolic answer engine with a small sequence-to-sequence or LLM-based reasoning module conditioned on the detected timeline (as structured context) plus the free-form question, to support paraphrased and compositional questions beyond the template set."),
      bullet("Validate on a real-world AQA benchmark (e.g. Clotho-AQA) once the pipeline's real-audio feature extraction is validated, to measure the sim-to-real gap directly."),

      H1("8. Reproducibility: File / Script Map"),
      table(
        ["Script", "Purpose"],
        [
          ["scripts/generate_audio_dataset.py", "Synthesizes the 220 audio clips + metadata.json (ground-truth timelines, environment labels)"],
          ["scripts/generate_qa_dataset.py", "Builds qa_pairs.json and the clip-level train/val/test split"],
          ["scripts/train_event_detector.py", "Extracts window features, trains the Random Forest event detector, saves model + confusion matrix + feature importance figures"],
          ["scripts/answer_qa.py", "Audio Context Layer inference: event detection + question parsing + symbolic answer engine"],
          ["scripts/evaluate_qa.py", "End-to-end test-set evaluation, per-type metrics, error analysis"],
          ["scripts/plot_learning_curve.py", "OOB error and learning curve figures (Section 6.3)"],
        ],
        [3600, 5400]
      ),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("/home/claude/audio_qa_poc/docs/Audio_Context_Layer_Technical_Report.docx", buf);
  console.log("Report written.");
});
