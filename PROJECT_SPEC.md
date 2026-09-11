# Project Specification (Original Design Document)

> This is the original design document PIShield-Lite was built from. Preserved here for reference. Implementation notes on where the final build matches or deviates from this spec are in `RESULTS.md`.

## Executive Summary

**Project Name:** PIShield-Lite (Internal Model Activation & Representation Analyzer)

**Primary Security Goal:** Detect both direct and indirect prompt injection attacks at zero extra generation latency by intercepting intermediate hidden states inside an open-weights LLM (such as Llama 3 8B or Mistral 7B) during the prefill phase.

## Core Theoretical Intuition

When an instruction-tuned LLM processes pure text, its residual stream represents static knowledge or conversational context. However, when an injected payload introduces imperative instructions (e.g., "Ignore above, instead perform..."), a distinct task shift and attention realignment occurs within the internal representations.

Rather than waiting for the LLM to output text or running an expensive external string-guardrail model, this architecture hooks directly into the Transformer's hidden layers. If the probe detects a "contaminated" internal activation vector during token prefilling, execution terminates immediately before tool parameters or unsafe text are generated.

## Complete System Architecture & Pipeline

```
[System Instruction + Untrusted Input]
                   │
                   ▼
     ┌───────────────────────────┐
     │ LLM Prefill Phase (Llama) │
     └─────────────┬─────────────┘
                   │  (Forward Pass - Layer N)
                   ▼
     ┌───────────────────────────┐
     │ Hidden State Extractor    │  <-- Hooks into Residual Stream Vector h_l
     └─────────────┬─────────────┘
                   │  (Vector h_l of Final Token)
                   ▼
     ┌───────────────────────────┐
     │ Lightweight Probe Probe   │  <-- Trained Logistic Regression / SVM Classifier
     └─────────────┬─────────────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
 [Clean: Probe = 0]    [Injected: Probe = 1]
       │                       │
       ▼                       ▼
 Continue Generation    ABORT & Intercept
  (Tool Call / Text)   (Throw Security Alert)
```

## Detailed Phase-by-Phase Method

### Phase 1: Training Data Construction & Feature Extraction

**Dataset Curation:**
- Clean Prompts (D_clean): System prompts concatenated with standard benign user inputs, RAG passages, and emails.
- Contaminated Prompts (D_injected): The same system prompts paired with untrusted data containing direct and indirect prompt injections (e.g., BIPIA, PromptBench, or bespoke RAG attack payloads).

**Layer-wise Activation Extraction (PyTorch / Hugging Face):**
- Register a PyTorch forward hook on the residual_stream (or `model.layers[i]`) across all L layers.
- Run a forward pass over D_clean and D_injected without calling `.generate()`.
- Extract the hidden activation vector h_l ∈ R^d corresponding to the final token of the prefill sequence at layer l ∈ [1, L].

### Phase 2: Probe Classifier Training & Critical Layer Selection

**Linear Separability Training:**
- For each layer l, train a simple binary linear classifier (Logistic Regression or Linear SVM) using the extracted hidden state vectors h_l and binary labels y ∈ {0, 1}.

**Critical Layer Identification:**
- Evaluate classification accuracy and Area Under the ROC Curve (AUC) on a 20% validation split.
- Select the Injection-Critical Layer (l*) — typically located in the mid-to-late transformer layers — that yields the highest separation between clean and contaminated representations.

### Phase 3: Runtime Inference Intercept (The Security Proxy)

- During live execution, pass the context string (System Prompt + Untrusted Data/User Input) to the LLM.
- Execute the prefill pass up to layer l*.
- Extract h_l* from the final prefill token.
- Predict p̂ = Probe(h_l*):
  - If p̂ ≥ τ (Threshold, e.g., 0.5): Immediately raise a `PromptInjectionException`, drop the context, and log an incident event.
  - If p̂ < τ: Allow the LLM to complete token generation safely.

## Tech Stack & Tools Required

- **Core Machine Learning & LLM Runtime:** Python 3.10+, PyTorch, Hugging Face `transformers`, `accelerate`
- **Target Open-Weights Models:** `meta-llama/Meta-Llama-3-8B-Instruct` or `mistralai/Mistral-7B-Instruct-v0.2`
- **Probe Modeling & Analysis:** scikit-learn (`LogisticRegression`, `SVC`, `SGDClassifier`), numpy
- **Data & Benchmarking Libraries:** `datasets`, Pandas, `evaluate`
- **Development Hardware / Environment:** Local macOS or Linux workspace with GPU acceleration (Apple Silicon MPS or CUDA)

## Concrete Implementation Blueprint (Original)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression

class PIShieldProbe:
    def __init__(self, model_id: str, critical_layer: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
        self.critical_layer = critical_layer
        self.classifier = LogisticRegression()
        self.extracted_activation = None

    def _hook_fn(self, module, input, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        self.extracted_activation = hidden_states[0, -1, :].detach().cpu().numpy()

    def extract_prefill_representation(self, text_prompt: str):
        inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.model.device)
        target_layer = self.model.model.layers[self.critical_layer]
        hook_handle = target_layer.register_forward_hook(self._hook_fn)
        with torch.no_grad():
            _ = self.model(**inputs)
        hook_handle.remove()
        return self.extracted_activation

    def fit_probe(self, X_train_activations, y_labels):
        self.classifier.fit(X_train_activations, y_labels)

    def predict_injection(self, text_prompt: str) -> bool:
        activation = self.extract_prefill_representation(text_prompt)
        prediction = self.classifier.predict([activation])
        return bool(prediction[0])
```

> **Note:** the final implementation in `pishield.py` deviates from this blueprint in two deliberate ways: (1) training and inference are separated — the classifier is trained offline and a frozen, pre-trained pipeline is passed into the class, matching real deployment practice; and (2) the hook uses an exception-based true early exit so layers after the critical layer are never computed, rather than letting the full forward pass run to completion. See `RESULTS.md` for the measured impact of this change (~19x latency improvement).

## Required Deliverables & Expected Outputs

1. **Layer-wise Activation Heatmap Report:** A comparative performance graph illustrating validation accuracy across all transformer layers (1…32), proving empirically which layer serves as the "Injection-Critical Layer" for the target LLM.
2. **Inference Performance Metrics:**
   - Accuracy Metrics: Target False Positive Rate (FPR) < 1% and False Negative Rate (FNR) < 3% across standard prompt injection benchmarks.
   - Latency Metrics: Proof of zero generation latency overhead, demonstrating that classification occurs in <15ms during the prompt prefill phase.
3. **Executable Security Intercept Module:** A Python library/wrapper that takes an open-weights model and context string, performs prefill activation probing, and raises a security exception prior to downstream execution.
4. **Adversarial Benchmark Script:** An evaluation runner testing the probe against direct jailbreaks (e.g., Do Anything Now) and indirect RAG injections (e.g., untrusted HTML/PDF text payloads).
