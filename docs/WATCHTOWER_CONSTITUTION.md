# WatchTower ML — System Constitution
**Version:** 2.0.0
**Author:** Geoffrey Kirumba — SCT212-0137/2022, JKUAT
**Module:** Machine Learning Detection Engine
**Last Updated:** 2026-06-01

---

## 1. Purpose

WatchTower ML is a machine learning-based Network Intrusion Detection System.
It accepts network flow features via a REST API, runs them through a three-model
ensemble, and returns structured detection results via the same API.

This constitution is the single source of truth for how the system is built,
what every component does, and how development must proceed. Every contributor
and every AI assistant working on this codebase must read this document before
writing or modifying any code.

---

## 2. Core Loop (What This System Does)

```
External caller (Postman / dashboard / network module)
        │
        │  POST /api/v1/detect/flow   ← single flow
        │  POST /api/v1/detect/batch  ← multiple flows
        │  POST /api/v1/detect/csv    ← CSV file upload
        ▼
┌─────────────────────────────────────────┐
│           FastAPI Layer                 │
│  Receives flow features as JSON/CSV     │
│  Returns detection results as JSON      │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│         Preprocessing Pipeline          │
│  Aligns features to trained schema      │
│  Scales values using saved pipeline     │
│  Converts to float32 numpy array        │
└──────────────────┬──────────────────────┘
                   │
         ┌─────────┼─────────┐
         ▼         ▼         ▼
      RF Model  XGBoost   Isolation
      predict   predict    Forest
      _proba()  _proba()  predict()
         │         │         │
         └─────────┼─────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│            Fusion Layer                 │
│  Combines RF + XGBoost predictions      │
│  Computes fused attack score            │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│         IF Escalation Layer             │
│  If RF+XGB say BENIGN but IF says       │
│  anomaly → override to ANOMALY (-1)     │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│         JSON Detection Result           │
│  Returned to caller via FastAPI         │
└─────────────────────────────────────────┘
```

---

## 3. File Structure (Canonical — Do Not Deviate)

```
watchtower-ml/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py              # App creation, lifespan, router registration
│   │   ├── dependencies.py      # ModelRegistry singleton — loads models once
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   └── detection.py     # All detection endpoints
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   └── detection.py     # Pydantic request/response models
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── detection_service.py    # Orchestrates full detection pipeline
│   │       └── ml_inference_service.py # Wraps model calls for the service layer
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── model_loader.py      # Load models, set runtime params, warmup
│   │   ├── predictor.py         # Input normalization, predict_proba, confidence
│   │   ├── fusion.py            # All fusion strategies (majority, or, weighted)
│   │   └── escalation.py        # IF escalation, attack scoring, threshold opt
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── capture_reader.py    # CSV and PCAP reading, label extraction
│   ├── preprocessing/
│   │   ├── flow_aggregator.py   # Groups packets into flows
│   │   ├── packet_capture.py    # Raw packet capture utilities
│   │   ├── preprocessing_pipeline1.py  # preprocess_for_inference()
│   │   └── custom_transformers.py      # Custom sklearn transformers
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── console_report.py    # All print/console output — no logic
│   ├── diagnostics/
│   │   └── schema_checker.py    # Feature alignment and model inspection
│   ├── models/                  # Saved model artifacts (never commit large files)
│   │   ├── random_forest.pkl
│   │   ├── xgboost_model.pkl
│   │   ├── isolation_forest.pkl
│   │   └── preprocessing_pipeline.pkl
│   ├── data/                    # Sample/test data only
│   ├── utils/
│   │   └── latency_tracker.py   # LatencyTracker — stdlib only
│   ├── main.py                  # Application entry point
│   └── run_detector.py          # CLI runner — thin orchestrator only
├── WATCHTOWER_CONSTITUTION.md   # This file
├── INTERFACE_CONTRACT.md        # API boundary agreement with external callers
├── requirements.txt
└── README.md
```

---

## 4. Module Responsibilities

Every module has exactly one job. If a module is doing two things,
it needs to be split. This is non-negotiable.

### 4.1 `api/main.py`
- Creates the FastAPI app
- Registers the lifespan (model loading at startup)
- Registers all routers
- Exposes root health check
- **Must NOT:** contain any ML logic, model loading, or business logic

### 4.2 `api/dependencies.py`
- Holds the `ModelRegistry` singleton
- Loads all three models exactly once at startup
- Sets runtime parameters (n_jobs, nthread, predictor)
- Runs model warmup after loading
- **Must NOT:** run inference, build DataFrames, touch routes

### 4.3 `api/routes/detection.py`
- Defines all HTTP endpoints
- Validates input via Pydantic schemas
- Calls service layer functions — nothing else
- Returns Pydantic response models
- **Must NOT:** contain ML logic, model loading, or direct model calls

### 4.4 `api/services/detection_service.py`
- Orchestrates the full detection pipeline for each request
- Calls preprocessing → inference → fusion → escalation in order
- Tracks latency via LatencyTracker
- Builds and returns the DetectionResult dict
- **Must NOT:** load models, define endpoints, or format console output

### 4.5 `api/services/ml_inference_service.py`
- Wraps individual model calls for the service layer
- Handles predict_proba calls and result extraction
- **Must NOT:** load models from disk, implement fusion, or touch API layer

### 4.6 `api/schemas/detection.py`
- Defines all Pydantic request and response models
- No logic — only type definitions
- **Must NOT:** import from inference, preprocessing, or service modules

### 4.7 `inference/model_loader.py`
- Loads serialized models from disk via joblib
- Configures runtime parameters after loading
- Resolves class names from model metadata
- Runs warmup inference after loading
- **Must NOT:** run real inference, build DataFrames, touch API layer

### 4.8 `inference/predictor.py`
- `prepare_input()` — converts any input type to float32 numpy array
- `predict_with_model()` — calls predict_proba and returns results
- `prediction_confidence()` — extracts per-sample confidence scores
- `class_name_for_prediction()` — maps index to label string
- **Must NOT:** load models, implement fusion, or call API

### 4.9 `inference/fusion.py`
- `combine_predictions()` — merges RF and XGBoost predictions
- `combine_scores()` — merges attack scores from both models
- Supports: majority, or, confidence_weighted, unanimous_or_majority
- **Must NOT:** load models, preprocess data, escalate, or call API

### 4.10 `inference/escalation.py`
- `apply_if_escalation()` — overrides BENIGN to ANOMALY when IF disagrees
- `attack_score_from_proba()` — computes non-BENIGN probability sum
- `optimize_threshold()` — finds best binary attack/benign threshold
- **Must NOT:** load models, implement fusion, or preprocess data

### 4.11 `ingestion/capture_reader.py`
- `csv_to_flow_features()` — reads CSV, extracts labels if present
- `pcap_to_flow_features()` — reads PCAP, returns flow DataFrame
- **Must NOT:** run inference, touch models, or call API

### 4.12 `utils/latency_tracker.py`
- `LatencyTracker` class — records stage timestamps and computes ms durations
- Stdlib only — no external dependencies
- **Must NOT:** import from any other project module

### 4.13 `reporting/console_report.py`
- `print_detection_report()` — formats and prints detection results to console
- Receives all data as arguments — computes nothing itself
- **Must NOT:** run inference, call API, or import from inference modules

---

## 5. Detection Pipeline Rules

### 5.1 Input
- All model inputs MUST be `np.float32` numpy arrays
- Conversion happens ONCE in `prepare_input()` — nowhere else
- Batch inference only — never call a model inside a per-flow Python loop
- `predict_proba()` is called ONCE per model per batch and reused —
  never call `predict()` and `predict_proba()` separately

### 5.2 Fusion
- Default fusion strategy is `majority`
- All four strategies are available for evaluation
- The fused result fed to escalation is always `majority` unless
  explicitly overridden via a request parameter

### 5.3 IF Escalation (non-negotiable rule)
```
IF fused_prediction == BENIGN (0)
AND if_prediction == anomaly (-1):
    final_prediction = -1   # ANOMALY
    escalated = True
```
This rule must never be removed, weakened, or made optional.
It is the primary safety net for zero-day threats.

### 5.4 IF Score Normalization
- Raw `score_samples()` output is normalized to [0, 1]
- 1.0 = most anomalous, 0.0 = most normal
- This normalized value is what the API and dashboard expose

---

## 6. API Contract

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Model status and feature counts |
| POST | `/api/v1/detect/flow` | Single flow detection |
| POST | `/api/v1/detect/batch` | Batch flow detection |
| POST | `/api/v1/detect/csv` | CSV file upload detection |

### Required Response Fields (every DetectionResult)

```json
{
  "flow_index": 0,
  "rf": {
    "prediction": 0,
    "label": "BENIGN",
    "confidence": 0.97
  },
  "xgb": {
    "prediction": 0,
    "label": "BENIGN",
    "confidence": 0.94
  },
  "isolation_forest": {
    "prediction": 1,
    "anomaly_score": 0.12,
    "escalated": false
  },
  "fused_prediction": 0,
  "fused_label": "BENIGN",
  "fused_attack_score": 0.03,
  "final_prediction": 0,
  "final_label": "BENIGN",
  "preprocessing_latency_ms": 12.4,
  "rf_inference_latency_ms": 45.2,
  "xgb_inference_latency_ms": 67.8,
  "if_inference_latency_ms": 38.1,
  "total_latency_ms": 163.5
}
```

### API Rules
- Models are NEVER loaded inside an endpoint — always from `ModelRegistry`
- No ML logic in `api/` — endpoints call service layer only
- All responses are JSON — no plain text, no HTML
- HTTP 422 for malformed input (Pydantic handles automatically)
- HTTP 500 for inference failures — include error detail in response
- CORS enabled for local dashboard consumption

---

## 7. Class Labels (Fixed — Do Not Change Without Retraining)

| Index | Label |
|---|---|
| 0 | BENIGN |
| 1 | Bot |
| 2 | DDoS |
| 3 | DoS GoldenEye |
| 4 | DoS Hulk |
| 5 | DoS Slowhttptest |
| 6 | DoS slowloris |
| 7 | FTP-Patator |
| 8 | PortScan |
| 9 | SSH-Patator |
| -1 | ANOMALY (IF escalation only) |

---

## 8. What Must Never Change Without Retraining

| Fixed Item | Location |
|---|---|
| Number of input features: 70 | `preprocessing_pipeline.pkl` |
| Feature names and order | `preprocessing_pipeline.pkl` |
| Scaler type and fitted values | `preprocessing_pipeline.pkl` |
| Class label indices 0–9 | `DEFAULT_CLASS_NAMES` in `model_loader.py` |
| IF contamination parameter | `isolation_forest.pkl` |

---

## 9. Performance Requirements

| Stage | Target | Hard Limit |
|---|---|---|
| Preprocessing | < 30ms | 100ms |
| RF inference | < 60ms | 150ms |
| XGBoost inference | < 80ms | 200ms |
| IF inference | < 50ms | 150ms |
| Total per flow | < 200ms | 500ms |
| API response time | < 250ms | 600ms |

Exceeding a hard limit must be logged as WARNING and investigated
before the next commit.

---

## 10. Modularization Rules (Mandatory)

These rules exist so the codebase remains understandable, maintainable,
and extensible — and so that Geoffrey always knows exactly where to look
when something breaks or needs changing.

### 10.1 File Length
- No file may exceed 300 lines
- If a file grows beyond 300 lines, it must be split before new
  features are added to it

### 10.2 Function Length
- No function may exceed 40 lines
- If a function is getting long, extract the inner logic into
  a named helper function with a clear docstring

### 10.3 One Responsibility Per Module
- Every module does exactly one thing (see Section 4)
- If you cannot describe a module's job in one sentence,
  it is doing too much and must be split

### 10.4 No Logic in Orchestrators
- `run_detector.py` and `api/routes/detection.py` contain
  orchestration only — they call other modules, never implement logic
- Any computation found in these files must be moved to the
  appropriate module immediately

### 10.5 No Cross-Layer Imports
- `api/` must not import from `inference/` directly —
  it goes through `services/`
- `inference/` modules must not import from each other
  except: `escalation.py` may import `attack_score_from_proba`
  from `predictor.py`
- `utils/` must not import from any other project module

### 10.6 Explicit Over Clever
- Prefer clear, readable code over compact or clever one-liners
- Variable names must describe what they hold:
  `proba1` not `p`, `fused_attack_score` not `fas`
- Every function must have a one-line docstring minimum

---

## 11. Explainability Rules (So Geoffrey Always Understands the Code)

These rules exist specifically to ensure the developer always
understands what every piece of code is doing and why.

### 11.1 Every Function Has a Docstring
```python
# Good
def apply_if_escalation(fused_preds, if_preds):
    """
    Override BENIGN predictions to ANOMALY where Isolation Forest
    detected anomalous behavior. Returns (final_preds, escalated_mask).
    """

# Bad
def apply_if_escalation(fused_preds, if_preds):
    ...
```

### 11.2 Non-obvious Logic Gets a Comment
```python
# Good
# score_samples() returns negative floats — flip so 1.0 = most anomalous
if_attack_scores = 1.0 - ((if_scores - min) / (max - min + 1e-9))

# Bad
if_attack_scores = 1.0 - ((if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-9))
```

### 11.3 Magic Numbers Are Named Constants
```python
# Good
EXPECTED_FEATURE_COUNT = 70
BENIGN_CLASS_INDEX = 0
IF_ANOMALY_LABEL = -1

# Bad
X = preprocess_for_inference(df, ..., expected_feature_count=70)
if final_preds[i] == 0 and if_preds[i] == -1:
```

### 11.4 AI-Generated Code Must Be Reviewed
- Never commit code generated by Copilot or any AI assistant
  without reading every line and understanding it
- If a line of generated code is not understood, ask before committing
- Add a comment above any non-trivial generated block explaining
  what it does in your own words

### 11.5 Log What Matters
Every major stage of the pipeline logs what it is doing:
```python
logger.info('Preprocessing %d flows...', len(df))
logger.info('RF inference complete: shape=%s dtype=%s', X.shape, X.dtype)
logger.info('IF escalation: %d flows escalated to ANOMALY', n_escalated)
```
Logs are how you debug the system when something goes wrong.
Silent failures are not acceptable.

---

## 12. Development Rules

1. **Never rewrite working code** — extend or wrap it
2. **One function, one job** — if it does two things, split it
3. **No logic in orchestrators** — `main()` and endpoints call modules only
4. **Batch always** — never call a model inside a per-flow loop
5. **All model inputs are float32** — enforced at `prepare_input()`
6. **Models load once** — `ModelRegistry` is the only place models load
7. **Test latency after every change** — regressions above 500ms are blockers
8. **Commit working state only** — never commit broken or half-finished code
9. **Constitution is updated when architecture changes** — not after
10. **If you do not understand a piece of code, do not ship it**

---

## 13. Integration Checklist

Run this before every submission or deployment:

- [ ] `GET /api/v1/health` returns `models_loaded: true`
- [ ] `POST /api/v1/detect/flow` returns a valid `DetectionResult`
- [ ] `POST /api/v1/detect/batch` with 100 flows completes under 500ms
- [ ] `POST /api/v1/detect/csv` processes a full CSV and returns results
- [ ] An escalated flow shows `isolation_forest.escalated: true`
- [ ] All latency fields are present and non-zero in every response
- [ ] No file in `src/` exceeds 300 lines
- [ ] No function in `src/` exceeds 40 lines
- [ ] Every function has a docstring
- [ ] `run_detector.py` CLI still works independently of the API
- [ ] All 10 class labels appear correctly in classification results

---

*This constitution is the single source of truth for WatchTower ML.
Any change that contradicts it requires an explicit decision and an
update to this document before the code change is made.*
