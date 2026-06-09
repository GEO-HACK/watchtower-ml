# Watchtower-ML — Comprehensive Project Documentation

> **Purpose:** Presentation-grade reference covering project architecture, all modules, the complete feature set, training pipeline, and all model evaluation metrics.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dataset](#3-dataset)
4. [Feature Engineering — Full Feature Set (70 Features)](#4-feature-engineering--full-feature-set-70-features)
5. [Data Preprocessing Pipeline](#5-data-preprocessing-pipeline)
6. [Model Training](#6-model-training)
   - [Random Forest](#61-random-forest)
   - [XGBoost](#62-xgboost)
   - [Isolation Forest](#63-isolation-forest)
7. [Model Evaluation Results & Metrics](#7-model-evaluation-results--metrics)
   - [Random Forest Results](#71-random-forest-results)
   - [XGBoost Results](#72-xgboost-results)
   - [Feature Importance (Random Forest)](#73-feature-importance-random-forest)
8. [Hybrid Inference Architecture](#8-hybrid-inference-architecture)
9. [Fusion Strategies](#9-fusion-strategies)
10. [Isolation Forest Escalation](#10-isolation-forest-escalation)
11. [Source Code Modules](#11-source-code-modules)
12. [REST API](#12-rest-api)
13. [Notebooks Index](#13-notebooks-index)
14. [Dependencies & Technology Stack](#14-dependencies--technology-stack)

---

## 1. Project Overview

**Watchtower-ML** is a hybrid network intrusion detection system (NIDS) that combines supervised machine learning with unsupervised anomaly detection to classify network traffic flows. The system is designed for real-time use, exposing a FastAPI service that:

- Accepts live flows from a companion signature-based module or uploaded CSV captures
- Runs three ML models in parallel (Random Forest, XGBoost, Isolation Forest)
- Fuses results using configurable strategies
- Escalates borderline cases using Isolation Forest anomaly scores
- Returns per-flow predictions with confidence scores and attack type labels

The project was trained on the **CICIDS 2017** dataset — a benchmark intrusion detection dataset containing 2.37 million labeled network flows spanning 10 traffic classes.

---

## 2. Repository Structure

```
watchtower-ml/
│
├── notebooks/                      # Training & exploration notebooks (Google Colab)
│   ├── combined_data.ipynb         # Data loading, EDA, preprocessing pipeline construction
│   ├── random_trainer.ipynb        # Random Forest training & evaluation
│   ├── xGboost.ipynb               # XGBoost training & evaluation
│   └── initial.ipynb               # Initial experiments
│
├── archive/                        # Legacy model artifacts and early notebooks
│   ├── feature_names (1).pkl
│   ├── label_encoder (1).pkl
│   ├── preprocessing_pipeline .pkl
│   ├── random_forest .pkl
│   └── notebooks/                  # Archived notebook copies
│
├── src/
│   ├── main.py                     # Application entry point
│   ├── run_detector.py             # CLI detector — loads models, runs inference, prints report
│   ├── run_hybrid_detection.py     # Full hybrid detection orchestrator with PCAP/CSV support
│   │
│   ├── api/
│   │   ├── app.py                  # FastAPI application with CORS and lifespan startup
│   │   ├── config.py               # Typed settings (model paths, feature count)
│   │   ├── schemas.py              # Pydantic I/O models
│   │   ├── routes/
│   │   │   └── detection.py        # /health, /detect (POST), /detect/live (GET) endpoints
│   │   └── services/
│   │       ├── detection_service.py    # Orchestrates ingestion → inference
│   │       └── ml_inference_service.py # Three-model inference + fusion + escalation
│   │
│   ├── inference/
│   │   ├── model_loader.py         # Load serialized models, warm-up, class name resolution
│   │   ├── predictor.py            # Input normalization, per-model inference calls
│   │   ├── fusion.py               # All multi-model fusion strategies
│   │   └── escalation.py          # Anomaly scoring, threshold optimization, IF escalation
│   │
│   ├── ingestion/
│   │   ├── csv_ingestion.py        # Load flows from CSV path or uploaded bytes
│   │   └── capture_reader.py       # PCAP and CSV flow feature extraction
│   │
│   ├── preprocessing/
│   │   ├── preprocessing_pipeline1.py   # Inference-time pipeline with validation
│   │   ├── custom_transformers.py       # FeatureAlignTransformer, OutlierClipper
│   │   ├── flow_aggregator.py           # Packet-level → flow-level feature aggregation
│   │   ├── packet_capture.py            # PCAP reading utilities
│   │   └── feature_names (1).pkl        # Serialized feature name list
│   │
│   ├── reporting/
│   │   └── console_report.py       # Full console detection report (predictions, metrics, latency)
│   │
│   ├── utils/
│   │   └── latency_tracker.py      # High-resolution stage latency tracker
│   │
│   ├── diagnostics/
│   │   └── schema_checker.py       # Feature schema validation utilities
│   │
│   └── models/                     # Serialized model artifacts
│       ├── random_forest .pkl      # Trained Random Forest classifier
│       ├── xgboost_model.pkl       # Trained XGBoost classifier
│       ├── isolation_forest.pkl    # Trained Isolation Forest
│       ├── preprocessing_pipeline .pkl   # Fitted preprocessing pipeline
│       └── feature_names.pkl       # Feature name registry
│
├── docs/
│   ├── SYSTEM_OVERVIEW.md          # Architecture diagrams and roadmap
│   ├── WATCHTOWER_CONSTITUTION.md  # Design principles
│   └── PROJECT_DOCUMENTATION.md   # This file
│
├── reports/
│   ├── schema_report.md            # Model compatibility and feature schema audit
│   └── WatchTower_Project_Progress_Report.docx
│
├── mis/                            # Sample misclassified flow PCAPs
│
├── test.ipynb                      # End-to-end inference test notebook
├── Untitled2.ipynb                 # Live inference experiments
├── diagnostics_preprocessing_leakage.py
├── rebuild_pipeline.py
├── rebuild_pipeline_correct.py
├── inspect_classes.py
└── requirements.txt
```

---

## 3. Dataset

### Source: CICIDS 2017 (Canadian Institute for Cybersecurity Intrusion Detection System)

| Property | Value |
|---|---|
| **Total files** | 6 CSV files (Monday–Friday captures) |
| **Combined raw rows** | 2,371,775 flows |
| **Raw columns** | 86 per file |
| **After deduplication & cleaning** | 2,371,704 flows, 75 columns |
| **After constant-column removal** | 75 → 70 model-ready features |

### CSV Files Loaded

| File | Description |
|---|---|
| `Monday-WorkingHours.pcap_ISCX.csv` | Normal traffic only |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | FTP-Patator, SSH-Patator |
| `Wednesday-workingHours.pcap_ISCX.csv` | DoS / Hulk / GoldenEye / Slowhttptest / Slowloris |
| `Friday-WorkingHours-Morning.pcap_ISCX.csv` | PortScan |
| `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` | PortScan |
| `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` | DDoS |

### Class Distribution (After Heartbleed Removal)

| Class | Label ID | Train Count | Train % | Test Count | Test % |
|---|---|---|---|---|---|
| BENIGN | 0 | 1,453,029 | 76.58% | 363,257 | 76.58% |
| DoS Hulk | 4 | 184,857 | 9.74% | 46,215 | 9.74% |
| PortScan | 8 | 127,144 | 6.70% | 31,786 | 6.70% |
| DDoS | 2 | 102,422 | 5.40% | 25,605 | 5.40% |
| DoS GoldenEye | 3 | 8,234 | 0.43% | 2,059 | 0.43% |
| FTP-Patator | 7 | 6,350 | 0.33% | 1,588 | 0.33% |
| DoS slowloris | 6 | 4,637 | 0.24% | 1,159 | 0.24% |
| SSH-Patator | 9 | 4,718 | 0.25% | 1,179 | 0.25% |
| DoS Slowhttptest | 5 | 4,399 | 0.23% | 1,100 | 0.23% |
| Bot | 1 | 1,573 | 0.08% | 393 | 0.08% |
| **Total** | — | **1,897,363** | **80%** | **474,341** | **20%** |

> **Heartbleed** was removed before training due to insufficient samples for reliable class learning.

> Stratified 80/20 split preserves exact class ratios across both sets (0.00% drift verified for all 10 classes).

---

## 4. Feature Engineering — Full Feature Set (70 Features)

The following 70 features are the model input after constant-column removal. These are derived from the CICFlowMeter tool and represent statistical summaries of bidirectional network flows.

### Removed Constant Columns (10 columns dropped from original 80)

These columns had zero variance across all samples and were excluded:

`Bwd PSH Flags`, `Fwd URG Flags`, `Bwd URG Flags`, `CWE Flag Count`,
`Fwd Avg Bytes/Bulk`, `Fwd Avg Packets/Bulk`, `Fwd Avg Bulk Rate`,
`Bwd Avg Bytes/Bulk`, `Bwd Avg Packets/Bulk`, `Bwd Avg Bulk Rate`

### Active Feature List

| # | Feature Name | Category |
|---|---|---|
| 1 | Source Port | Connection |
| 2 | Destination Port | Connection |
| 3 | Protocol | Connection |
| 4 | Flow Duration | Flow Timing |
| 5 | Total Fwd Packets | Volume |
| 6 | Total Backward Packets | Volume |
| 7 | Total Length of Fwd Packets | Volume |
| 8 | Total Length of Bwd Packets | Volume |
| 9 | Fwd Packet Length Max | Packet Size — Forward |
| 10 | Fwd Packet Length Min | Packet Size — Forward |
| 11 | Fwd Packet Length Mean | Packet Size — Forward |
| 12 | Fwd Packet Length Std | Packet Size — Forward |
| 13 | Bwd Packet Length Max | Packet Size — Backward |
| 14 | Bwd Packet Length Min | Packet Size — Backward |
| 15 | Bwd Packet Length Mean | Packet Size — Backward |
| 16 | Bwd Packet Length Std | Packet Size — Backward |
| 17 | Flow Bytes/s | Throughput |
| 18 | Flow Packets/s | Throughput |
| 19 | Flow IAT Mean | Inter-Arrival Time |
| 20 | Flow IAT Std | Inter-Arrival Time |
| 21 | Flow IAT Max | Inter-Arrival Time |
| 22 | Flow IAT Min | Inter-Arrival Time |
| 23 | Fwd IAT Total | Inter-Arrival Time — Forward |
| 24 | Fwd IAT Mean | Inter-Arrival Time — Forward |
| 25 | Fwd IAT Std | Inter-Arrival Time — Forward |
| 26 | Fwd IAT Max | Inter-Arrival Time — Forward |
| 27 | Fwd IAT Min | Inter-Arrival Time — Forward |
| 28 | Bwd IAT Total | Inter-Arrival Time — Backward |
| 29 | Bwd IAT Mean | Inter-Arrival Time — Backward |
| 30 | Bwd IAT Std | Inter-Arrival Time — Backward |
| 31 | Bwd IAT Max | Inter-Arrival Time — Backward |
| 32 | Bwd IAT Min | Inter-Arrival Time — Backward |
| 33 | Fwd PSH Flags | TCP Flags |
| 34 | Fwd Header Length | Header Metrics |
| 35 | Bwd Header Length | Header Metrics |
| 36 | Fwd Packets/s | Throughput |
| 37 | Bwd Packets/s | Throughput |
| 38 | Min Packet Length | Packet Size — Global |
| 39 | Max Packet Length | Packet Size — Global |
| 40 | Packet Length Mean | Packet Size — Global |
| 41 | Packet Length Std | Packet Size — Global |
| 42 | Packet Length Variance | Packet Size — Global |
| 43 | FIN Flag Count | TCP Flags |
| 44 | SYN Flag Count | TCP Flags |
| 45 | RST Flag Count | TCP Flags |
| 46 | PSH Flag Count | TCP Flags |
| 47 | ACK Flag Count | TCP Flags |
| 48 | URG Flag Count | TCP Flags |
| 49 | ECE Flag Count | TCP Flags |
| 50 | Down/Up Ratio | Flow Symmetry |
| 51 | Average Packet Size | Packet Size — Global |
| 52 | Avg Fwd Segment Size | Segment Size |
| 53 | Avg Bwd Segment Size | Segment Size |
| 54 | Fwd Header Length.1 | Header Metrics |
| 55 | Subflow Fwd Packets | Subflow |
| 56 | Subflow Fwd Bytes | Subflow |
| 57 | Subflow Bwd Packets | Subflow |
| 58 | Subflow Bwd Bytes | Subflow |
| 59 | Init_Win_bytes_forward | TCP Window |
| 60 | Init_Win_bytes_backward | TCP Window |
| 61 | act_data_pkt_fwd | Active Data |
| 62 | min_seg_size_forward | Segment Size |
| 63 | Active Mean | Activity |
| 64 | Active Std | Activity |
| 65 | Active Max | Activity |
| 66 | Active Min | Activity |
| 67 | Idle Mean | Idle Time |
| 68 | Idle Std | Idle Time |
| 69 | Idle Max | Idle Time |
| 70 | Idle Min | Idle Time |

---

## 5. Data Preprocessing Pipeline

### Pipeline Steps

```
FeatureAlignTransformer  →  SimpleImputer  →  (OutlierClipper)  →  RobustScaler
```

| Step | Class | Purpose |
|---|---|---|
| `feature_aligner` | `FeatureAlignTransformer` | Ensures exact column order and names match training schema; fills missing columns with NaN |
| `imputer` | `SimpleImputer(strategy='median')` | Fills NaN values (including from infinity replacements) with column medians |
| `clipper` | `OutlierClipper(factor=10.0)` | Clips values beyond 10× IQR to prevent scale explosion |
| `scaler` | `RobustScaler` | Scales features using median and IQR — robust to outliers |

### Data Cleaning Steps (Applied Before Pipeline)

1. Strip whitespace from column names
2. Replace `+inf` / `-inf` with `NaN`
3. Drop columns with >50% missing values (0 dropped)
4. Drop constant columns — zero variance (10 dropped)
5. Drop exact duplicate rows (60 rows removed)
6. Remove `source_file` tracking column

### Fitting Strategy

The pipeline was fitted on a 100,000-sample stratified subset of the training data to limit RAM usage during fitting. Full data was then transformed in 50,000-row chunks.

### Column Name Mapping (PCAP → CICIDS)

The inference pipeline includes a bidirectional mapping that translates CICFlowMeter snake_case column names (used in live PCAP extraction) to the CICIDS Title Case column names used during training. Examples:

| PCAP Name | CICIDS Name |
|---|---|
| `src_port` | Source Port |
| `dst_port` | Destination Port |
| `flow_bytes_per_s` | Flow Bytes/s |
| `init_win_bytes_forward` | Init Win bytes forward |
| `fin_flag_count` | FIN Flag Count |

---

## 6. Model Training

### 6.1 Random Forest

**File:** `notebooks/random_trainer.ipynb`

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | None (unlimited) |
| `min_samples_split` | 10 |
| `min_samples_leaf` | 5 |
| `max_features` | `'sqrt'` |
| `class_weight` | `'balanced_subsample'` |
| `random_state` | 42 |
| `n_jobs` | -1 (all cores) |

Training data shape at inference: **(1,725,584 × 68)** — note: a slightly smaller version of the dataset was used due to pipeline version differences between early training runs.

### 6.2 XGBoost

**File:** `notebooks/xGboost.ipynb`

| Hyperparameter | Value |
|---|---|
| `n_estimators` | 300 |
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `min_child_weight` | 5 |
| `gamma` | 0.1 |
| `reg_alpha` | 0.1 |
| `reg_lambda` | 1.0 |
| `objective` | `multi:softmax` |
| `num_class` | 10 |
| `eval_metric` | `mlogloss` |
| `random_state` | 42 |
| `n_jobs` | -1 |

**Class Imbalance Handling:** `compute_sample_weight(class_weight='balanced')` applied during `fit()`.

Sample weight range: **0.1306 → 120.6207**

Training data shape: **(1,897,363 × 70)**

### 6.3 Isolation Forest

**File:** `src/models/isolation_forest.pkl`

The Isolation Forest is trained as an unsupervised anomaly detector on the same feature set. It is not used for class labeling but instead provides a second opinion — flows that both supervised models classify as BENIGN but the Isolation Forest flags as anomalous are **escalated** to `ANOMALY` status.

Score normalization at inference:

```
if_attack_score = 1.0 - ( (raw_score - min) / (max - min + 1e-9) )
```

Higher `if_attack_score` = more anomalous.

---

## 7. Model Evaluation Results & Metrics

### 7.1 Random Forest Results

**Test set shape: (431,397 × 68)**

#### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| BENIGN (0) | 1.00 | 1.00 | 1.00 | 346,659 |
| Bot (1) | 0.70 | 0.97 | **0.81** | 390 |
| DDoS (2) | 1.00 | 1.00 | 1.00 | 25,603 |
| DoS GoldenEye (3) | 0.99 | 1.00 | 0.99 | 2,057 |
| DoS Hulk (4) | 1.00 | 1.00 | 1.00 | 34,570 |
| DoS Slowhttptest (5) | 1.00 | 0.99 | 0.99 | 1,046 |
| DoS slowloris (6) | 0.99 | 0.99 | 0.99 | 1,077 |
| FTP-Patator (7) | 1.00 | 1.00 | 1.00 | 1,187 |
| PortScan (8) | 1.00 | 1.00 | 1.00 | 18,164 |
| SSH-Patator (9) | 1.00 | 1.00 | 1.00 | 644 |
| **macro avg** | **0.97** | **0.99** | **0.98** | 431,397 |
| **weighted avg** | **1.00** | **1.00** | **1.00** | 431,397 |
| **Overall Accuracy** | — | — | **1.00** | — |

**Overall Macro F1: 0.9791**

#### Bot Class Deep-Dive (Hardest Class)

| Metric | Value |
|---|---|
| Total Bot test samples | 390 |
| Correctly identified (True Positives) | 378 |
| Missed as BENIGN (False Negatives) | 12 |
| False Positives (BENIGN → Bot) | 161 |

*Observation:* Bot traffic shares strong feature overlap with BENIGN flows (similar packet sizes, inter-arrival times). The 12 false negatives and 161 false positives all cluster around the same feature region in the scaled space, indicating this class boundary is genuinely ambiguous in the CICIDS feature space.

---

### 7.2 XGBoost Results

**Test set shape: (474,341 × 70)**

#### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| BENIGN | 1.0000 | 0.9996 | **0.9998** | 363,257 |
| Bot | 0.8846 | 0.9949 | **0.9365** | 393 |
| DDoS | 0.9996 | 0.9999 | 0.9997 | 25,605 |
| DoS GoldenEye | 0.9956 | 0.9995 | 0.9976 | 2,059 |
| DoS Hulk | 0.9988 | 0.9998 | 0.9993 | 46,215 |
| DoS Slowhttptest | 0.9937 | 1.0000 | 0.9968 | 1,100 |
| DoS slowloris | 0.9974 | 0.9974 | 0.9974 | 1,159 |
| FTP-Patator | 0.9994 | 1.0000 | 0.9997 | 1,588 |
| PortScan | 0.9999 | 0.9999 | 0.9999 | 31,786 |
| SSH-Patator | 1.0000 | 1.0000 | **1.0000** | 1,179 |
| **macro avg** | **0.9869** | **0.9991** | **0.9927** | 474,341 |
| **weighted avg** | **0.9997** | **0.9997** | **0.9997** | 474,341 |
| **Overall Accuracy** | — | — | **0.9997** | — |

**Overall Macro F1: 0.9927**

---

### 7.3 Model Comparison Summary

| Metric | Random Forest | XGBoost |
|---|---|---|
| Overall Accuracy | ~1.00 | 0.9997 |
| Macro F1 | **0.9791** | **0.9927** |
| Weighted F1 | 1.00 | 0.9997 |
| Bot F1 | 0.81 | **0.9365** |
| DDoS F1 | 1.00 | 0.9997 |
| PortScan F1 | 1.00 | 0.9999 |
| SSH-Patator F1 | 1.00 | 1.0000 |
| FTP-Patator F1 | 1.00 | 0.9997 |

> XGBoost outperforms Random Forest on all minority classes, especially the **Bot** class (+12.6 F1 points). Both models are used together in the hybrid system to maximize coverage.

---

### 7.4 Feature Importance (Random Forest — Top 20)

| Rank | Feature | Importance Score |
|---|---|---|
| 1 | Destination Port | 0.08369 |
| 2 | Bwd Header Length | 0.03611 |
| 3 | Bwd Packets/s | 0.03232 |
| 4 | Init_Win_bytes_forward | 0.02803 |
| 5 | Init_Win_bytes_backward | 0.02796 |
| 6 | Packet Length Mean | 0.02606 |
| 7 | Average Packet Size | 0.02568 |
| 8 | Avg Fwd Segment Size | 0.02275 |
| 9 | Fwd IAT Mean | 0.02272 |
| 10 | Fwd Packet Length Max | 0.02222 |
| 11 | Subflow Bwd Packets | 0.02189 |
| 12 | Avg Bwd Segment Size | 0.02138 |
| 13 | Subflow Bwd Bytes | 0.02137 |
| 14 | Max Packet Length | 0.02063 |
| 15 | Flow IAT Max | 0.02006 |
| 16 | Flow Packets/s | 0.02005 |
| 17 | Flow IAT Mean | 0.01971 |
| 18 | Total Length of Bwd Packets | 0.01959 |
| 19 | Bwd Packet Length Mean | 0.01919 |
| 20 | min_seg_size_forward | 0.01896 |

**Key observation:** `Destination Port` is by far the most important single feature (8.37%), reflecting that many attack types target well-known ports. TCP window initialization features (`Init_Win_bytes_*`) are highly informative for distinguishing attack behavior at the TCP handshake level.

---

## 8. Hybrid Inference Architecture

The system runs three models on every incoming flow batch and fuses results through a layered decision process:

```
                    ┌──────────────────────────────┐
   CSV / PCAP  ───► │   Ingestion + Preprocessing  │
                    └──────────────┬───────────────┘
                                   │ 70-feature float32 matrix
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                     ▼
    ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
    │  Random Forest   │  │    XGBoost       │  │ Isolation Forest │
    │  (Supervised)    │  │  (Supervised)    │  │ (Unsupervised)   │
    └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
             │ preds1/proba1        │ preds2/proba2        │ if_preds/if_scores
             └──────────┬──────────┘                      │
                        ▼                                  │
              ┌──────────────────────┐                     │
              │   Fusion Layer       │◄────────────────────┘
              │  (4 strategies)      │   IF Escalation
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │   Final Predictions  │
              │  label / attack_type │
              │  confidence / scores │
              └──────────────────────┘
```

### Per-Flow Output Schema

Each flow produces the following prediction object (`PredictionResult`):

| Field | Type | Description |
|---|---|---|
| `flow_id` | string | Auto-constructed identifier: `src:port → dst:port/proto@time` |
| `label` | string | `"Normal"` or `"Attack"` |
| `attack_type` | string \| null | Attack class name (e.g. `"DDoS"`) or null if benign |
| `confidence` | float [0–1] | Fused confidence score |
| `model1_label` | string | Random Forest class name |
| `model1_confidence` | float | RF max class probability |
| `model2_label` | string | XGBoost class name |
| `model2_confidence` | float | XGB max class probability |
| `if_prediction` | int | `+1` = normal, `-1` = anomaly |
| `if_anomaly_score` | float | Normalized anomaly score (1.0 = most anomalous) |
| `fused_label` | string | Output of majority fusion |
| `fused_score` | float | Attack probability from fusion layer |
| `escalated` | bool | True if IF overrode a BENIGN to ANOMALY |
| `models_agree` | bool | True if RF and XGBoost agree |

---

## 9. Fusion Strategies

All four strategies are computed on every request. The **majority** strategy is used for the final decision:

| Strategy | Logic |
|---|---|
| `majority` | If both models agree → use that label; if disagree → RF takes precedence |
| `or` | Any model flagging as attack → report as attack (most sensitive) |
| `confidence_weighted` | Whichever model has higher per-sample max-probability wins |
| `unanimous_or_majority` | Agreement → unanimous label; disagreement → highest non-zero label |

**Attack Score Calculation** (per fusion strategy):

- Score = sum of all non-BENIGN class probabilities from the model output
- When both models agree: score = average of both model scores
- When only one model flags attack: use that model's score alone
- On disagreement: take the maximum of both scores

---

## 10. Isolation Forest Escalation

The Isolation Forest acts as a safety net for zero-day and novel threats not seen during supervised training:

**Escalation Logic:**
```
IF (RF says BENIGN) AND (XGBoost says BENIGN) AND (Isolation Forest says ANOMALY):
    → Override final label to ANOMALY (-1)
    → Set escalated = True
```

This design ensures that:
- Known attack patterns are labeled with their specific class (DDoS, Bot, etc.)
- Novel anomalies that evade both supervised models are still flagged
- Escalation never degrades a confirmed attack detection

---

## 11. Source Code Modules

### `src/api/app.py`
FastAPI application. On startup, waits up to 10×3s for the companion signature module at `FLOW_API_URL` before accepting traffic. Mounts CORS middleware for frontend on `localhost:5173`.

### `src/api/routes/detection.py`
Three endpoints:
- `GET /health` — service liveness check
- `GET /detect/live` — pulls flows from signature module → runs inference → pushes results back to fusion layer
- `POST /detect` — accepts uploaded CSV or local path → runs inference

### `src/api/services/detection_service.py`
Singleton service (double-checked locking) that owns the `MLInferenceService` instance. Translates ingestion results and exceptions to HTTP-safe error types.

### `src/api/services/ml_inference_service.py`
Loads all three models on first request. Orchestrates the full inference pipeline:
1. `prepare_input()` — normalizes raw DataFrame
2. `preprocess_for_inference()` — aligned + imputed + scaled
3. RF `predict_proba()` → `preds1`, `proba1`
4. XGB `predict_proba()` → `preds2`, `proba2`
5. IF `predict()` + `score_samples()` → `if_preds`, `if_scores`
6. Four-strategy fusion via `combine_predictions()` / `combine_scores()`
7. IF escalation pass
8. Build per-flow result dicts

### `src/inference/model_loader.py`
- `load_maybe_dict_model()` — handles both raw model files and `{"model": ..., "pipeline": ..., "encoder": ...}` dict bundles
- `get_class_names()` — resolves integer class indices to string names
- `warmup_models()` — runs one dummy inference on each model to prime native thread pools

### `src/inference/predictor.py`
- `prepare_input()` — accepts dict / list / Series / DataFrame / ndarray; returns float32 array or DataFrame for pipeline consumption
- `predict_with_model()` — runs `predict()` and optionally `predict_proba()`
- `prediction_confidence()` — extracts per-sample confidence from probability matrix

### `src/inference/fusion.py`
Pure functions for all four fusion strategies (`combine_predictions`, `combine_scores`). No model dependencies.

### `src/inference/escalation.py`
- `attack_score_from_proba()` — sum of non-BENIGN probabilities
- `optimize_threshold()` — grid search over candidate thresholds maximizing binary F1 against ground truth
- `apply_if_escalation()` — applies the IF override rule

### `src/preprocessing/preprocessing_pipeline1.py`
Inference-time pipeline loader with strict validation:
- Validates pipeline has all 4 steps: `feature_aligner`, `imputer`, `clipper`, `scaler`
- Checks feature count matches `expected_feature_count` (70)
- Confirms imputer and scaler are fitted
- Validates model feature count matches pipeline schema
- Applies PCAP→CICIDS column name translation

### `src/preprocessing/custom_transformers.py`
Contains:
- `FeatureAlignTransformer` — sklearn-compatible transformer for column alignment; importable from a stable module path so serialized pipelines load correctly outside notebooks
- `OutlierClipper` — IQR-based outlier capping

### `src/ingestion/csv_ingestion.py`
Thin wrapper over `capture_reader.csv_to_flow_features()`. Adds debug sampling (100k rows max) to prevent memory spikes during development.

### `src/reporting/console_report.py`
Produces the full terminal detection report:
- Per-model prediction distributions
- Model agreement percentage
- Per-strategy fusion results with accuracy and F1 (when ground truth available)
- Isolation Forest escalation stats
- Sample flow details (first 10 flows)
- Latency breakdown by stage

### `src/utils/latency_tracker.py`
High-resolution (`time.perf_counter()`) stage timer that tracks `preprocessing`, `rf_inference`, `xgb_inference`, `if_inference`, and computes total.

---

## 12. REST API

### Base URL: `http://localhost:8001` (ML module)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Returns service name, status, and UTC timestamp |
| `/detect` | POST | Upload CSV file or provide path query param; returns batch predictions |
| `/detect/live` | GET | Pulls live flows from signature module and returns predictions |

### Response Structure (`DetectResponse`)

```json
{
  "status": "success",
  "processing_status": "completed",
  "timestamp": "2026-06-09T14:00:00Z",
  "prediction": { ... },
  "predictions": [ { "flow_id": "...", "label": "Attack", "attack_type": "DDoS", "confidence": 0.9987, ... } ],
  "meta": {
    "flows_processed": 1234,
    "processing_latency_ms": {
      "preprocessing": 12.34,
      "rf_inference": 45.67,
      "xgb_inference": 23.45,
      "if_inference": 8.90,
      "total": 90.36
    }
  }
}
```

### Integration with Signature Module
After inference, the ML module **pushes predictions** to the signature module's fusion endpoint (`SIGNATURE_FUSION_URL = localhost:8000/flows/ml-results`) so the two detection layers can cross-reference results.

---

## 13. Notebooks Index

| Notebook | Location | Purpose | Key Outputs |
|---|---|---|---|
| `combined_data.ipynb` | `notebooks/` | Full EDA, cleaning, pipeline construction | 2.37M row merged dataset; fitted pipeline; saved `.npy` arrays |
| `random_trainer.ipynb` | `notebooks/` | RF training, evaluation, Bot analysis | RF model, confusion matrix, F1 chart, feature importance |
| `xGboost.ipynb` | `notebooks/` | XGBoost training & evaluation | XGB model (Macro F1: 0.9927) |
| `test.ipynb` | root | End-to-end inference test on raw CSV | Validation of loaded models on Friday DDoS file |
| `Untitled2.ipynb` | root | Live pipeline and model debugging | Confirmed model output schema and column alignment |
| `initial.ipynb` | `notebooks/` | First experiments | Initial feasibility checks |

---

## 14. Dependencies & Technology Stack

### Core ML

| Library | Version | Role |
|---|---|---|
| scikit-learn | 1.6.1 | Random Forest, Isolation Forest, preprocessing pipeline |
| xgboost | 3.2.0 | XGBoost classifier |
| imbalanced-learn | 0.14.1 | SMOTE (explored during EDA) |
| numpy | 2.4.3 | Numerical arrays |
| pandas | 3.0.1 | Tabular data manipulation |
| joblib | 1.5.3 | Model serialization |

### API & Web

| Library | Version | Role |
|---|---|---|
| fastapi | 0.135.2 | REST API framework |
| uvicorn | 0.42.0 | ASGI server |
| httpx | 0.28.1 | Async HTTP client (signature module calls) |
| pydantic | 2.12.5 | Schema validation and serialization |
| python-multipart | 0.0.20 | File upload support |

### Data & Visualization

| Library | Version | Role |
|---|---|---|
| matplotlib | 3.10.8 | Training charts (confusion matrix, F1 bars, feature importance) |
| seaborn | 0.13.2 | EDA heatmaps |
| pyarrow | 23.0.1 | Fast CSV I/O |

### Development

| Library | Version | Role |
|---|---|---|
| jupyterlab | 4.5.6 | Notebook execution |
| scipy | 1.17.1 | Statistical utilities |

---

*Generated: 2026-06-09 | Branch: main | Watchtower-ML v1.0.0*
