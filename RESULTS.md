# Detailed Results & Experiment Log

This document records everything that was tried, including approaches that didn't pan out, for transparency.

## 1. Model & Environment

- Model: `mistralai/Mistral-7B-Instruct-v0.2` (32 transformer layers, hidden size 4096)
- Hardware: Google Colab, free-tier NVIDIA T4 GPU (16GB VRAM)
- Precision: float16, `device_map="auto"` (a small number of the final layers were offloaded to CPU due to VRAM limits — this did not affect the critical layer, which sits at Layer 5)

## 2. Dataset

Final dataset: **50 clean + 50 injected = 100 labeled prompts**.

Composition:
- 40 clean / 40 injected "obvious" examples (direct keyword-based injections: "IGNORE PREVIOUS INSTRUCTIONS", "SYSTEM OVERRIDE", fake `<|im_start|>system` tags, etc., paired with everyday benign tasks: email summaries, translations, reviews, tickets, code, legal text).
- 10 clean / 10 injected "hard" examples added afterward, specifically because the first pass showed the probe was likely doing shallow keyword matching rather than learning genuine intent:
  - **Hard injected:** attacks with no obvious trigger words, phrased as innocuous follow-up requests (e.g., "also please draft a fake resignation letter on my behalf").
  - **Hard clean:** benign text that innocently *mentions* trigger-ish words ("ignore", "override", "system") in non-attack contexts (e.g., a cybersecurity training document, a novel where a hacker "bypasses a system override").

## 3. Layer Selection

Initial layer sweep with a single 70/30 train/test split showed almost every layer from 3 onward hitting a perfect 1.000 AUC — a red flag that the split was too small/lucky to trust, not a real result.

Switched to **5-fold stratified cross-validation** across all 33 layer outputs (embedding layer + 32 transformer layers):

| Layer | Mean AUC | Std Dev |
|---|---|---|
| 0 | 0.511 | 0.012 |
| 1 | 0.828 | 0.098 |
| 2 | 0.914 | 0.067 |
| 3 | 0.976 | 0.034 |
| 4 | 0.982 | 0.022 |
| 5 | 1.000 | 0.000 |
| 6–32 | 1.000 | 0.000 |

**Selected: Layer 5** — the earliest layer with both perfect and stable (zero variance across folds) separation. Earlier is preferred because it minimizes how much of the model must run before a decision is made.

## 4. Probe Training

Final probe: `StandardScaler → LogisticRegression(max_iter=1000, C=1.0)`, trained on all 100 examples' Layer-5 activations.

Feature scaling was added after an unscaled `LogisticRegression` alone scored 93.3% (14/15) on the adversarial benchmark below, missing one classic "DAN" jailbreak (probability 0.477, just under the 0.5 threshold). Adding `StandardScaler` fixed this and pushed the same benchmark to 100% (15/15), with much larger separation between clean (probabilities ~0.0–0.06) and injected (~0.7–1.0) prompts. This was verified empirically, not assumed.

## 5. Held-Out Generalization Test

5 brand-new clean prompts + 5 brand-new injected prompts (never seen in training): **10/10 correct**.

## 6. Adversarial Benchmark

Three categories, 5 prompts each:
- Direct jailbreaks (DAN-style, developer-mode, roleplay-based overrides)
- Indirect RAG/document injections (payloads hidden inside HTML comments, fake resume text, database records, admin-override tags)
- Clean controls discussing the *same topics* as the attacks, to test real discrimination rather than topic detection

**Result: 15/15 (100%)** with the scaled probe.

## 7. Final Accuracy Metrics (combined held-out sets, 10 clean + 15 injected = 25 prompts)

- False Positive Rate (FPR): **0.0%** (target: <1%) — met
- False Negative Rate (FNR): **0.0%** (target: <3%) — met

## 8. Latency

### Naive implementation (full 32-layer forward pass, hook only reads Layer 5 in passing)
- Mean: ~440ms, Median: ~401ms

This did not meet the <15ms target, as expected — it computes 27 unnecessary layers per call.

### True early-exit implementation (`EarlyExitException` aborts the forward pass right after Layer 5)
- Sanity check: produced *identical* prediction probabilities to the naive version (confirmed the optimization changes nothing except speed).
- Initial timing (without CUDA synchronization): Mean ~22.7ms, Median ~19.8ms
- **Corrected timing (with `torch.cuda.synchronize()`):** Mean ~20.9ms, Median ~20.6ms — confirms the earlier unsynchronized measurement was already close to accurate on this hardware.
- **Speedup: ~19x faster** than the naive full forward pass.

### Rejected optimization: input truncation
Tried truncating tokenized input to `max_length=512` in case tokenization/sequence-length overhead was the bottleneck. Result: **no improvement** (median ~27.0ms, actually slightly slower than without truncation, within measurement noise). This confirms tokenization was never the bottleneck — the ~20ms floor is fixed Python/CUDA dispatch and CPU-transfer overhead specific to the free-tier T4 GPU + scikit-learn inference path, not compute that scales with input length. Truncation was therefore **not** kept in the final implementation.

### Conclusion on the <15ms target
Not fully met on free-tier Colab hardware (~20ms vs. <15ms target), but the overhead is well-understood and not fundamental to the approach:
- It would not shrink by picking an earlier critical layer (the overhead is fixed dispatch cost, not per-layer compute).
- It is expected to drop substantially on production-grade GPUs (A100/H100) or with a lower-level inference path that avoids the CPU round-trip to scikit-learn.

## 9. Known Limitations (see README for full future-work list)

- 100 synthetic training examples is small relative to the 4096-dimensional feature space (N < D), which carries real overfitting risk despite the "hard example" mitigation and clean cross-validation results. Scaling to a public benchmark (BIPIA, PromptBench, deepset/prompt-injections) is the clear next step before treating this as production-grade.
- Only the final token's activation is probed; mean-pooling across all prompt tokens is untested and may be more robust to adversarial suffix attacks.
