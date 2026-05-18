# Real-Time Streaming with Apache Kafka
### ENGR 5785G — Assignment 1

A complete, three-component pipeline that streams Bike Sharing sensor records
through Apache Kafka, classifies demand in real time using a pre-trained
Random Forest model via the **Faust Streams API**, and prints live predictions
to a consumer terminal.

---

## Dataset

| Field | Value |
|---|---|
| Source | UCI Bike Sharing Dataset (archive.ics.uci.edu/dataset/275) |
| Records | 17 379 hourly observations (2011–2012) |
| Features | season, year, month, hour, holiday, weekday, working day, weather, temp, humidity, wind |
| Target | Demand class: **Low** / **Medium** / **High** (derived from hourly rental count) |

> A synthetic dataset generator (`data/generate_data.py`) that mirrors the UCI
> statistics is included so the pipeline runs out-of-the-box without a manual
> download.

---

## ML Model

| Item | Value |
|---|---|
| Algorithm | Random Forest Classifier (200 trees, max_depth=12) |
| Task | 3-class demand classification (Low / Medium / High) |
| Library used | scikit-learn |
| Saved as | `model/bike_model.joblib` |
| **Accuracy** | **0.8826 (88.26 %)** |
| **F1 (macro)** | **0.8838** |

Class thresholds are derived automatically from the 33rd and 66th percentiles
of the training data and stored inside the model bundle.

---

## Streams Library

**Option A — Python + Faust**

```
faust -A streams_processor worker -l info
```

`streams_processor.py` defines a `@app.agent(raw_topic)` that consumes each
`RawRecord`, builds the feature vector, runs `model.predict()` synchronously
(< 1 ms per row), and publishes a `PredictionRecord` to the `predictions` topic.
No hand-rolled consumer loop is used anywhere.

---

## Project Structure

```
Kafka/
├── data/
│   ├── generate_data.py        # Synthetic dataset generator
│   └── bike_sharing.csv        # Auto-created on first run
├── model/
│   └── bike_model.joblib       # Serialised Random Forest + metadata
├── producer.py                 # Streams rows → raw-data topic  (~1 row/s)
├── streams_processor.py        # Faust agent: raw-data → ML → predictions
├── consumer.py                 # Pretty-prints predictions topic
├── train_model.py              # Offline training script
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.11+
- A running Kafka broker — either **local** (Docker recommended) or
  **Confluent Cloud**

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Kafka (local — Docker Compose)

```bash
docker run -d --name kafka \
  -p 9092:9092 \
  -e KAFKA_ENABLE_KRAFT=yes \
  -e KAFKA_CFG_NODE_ID=1 \
  -e KAFKA_CFG_PROCESS_ROLES=broker,controller \
  -e KAFKA_CFG_CONTROLLER_QUORUM_VOTERS=1@localhost:9093 \
  -e KAFKA_CFG_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093 \
  -e KAFKA_CFG_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
  -e KAFKA_CFG_CONTROLLER_LISTENER_NAMES=CONTROLLER \
  bitnami/kafka:latest
```

### 3. Confluent Cloud (optional)

Export your credentials before running any component:

```bash
export KAFKA_BOOTSTRAP_SERVERS="pkc-xxxxx.region.aws.confluent.cloud:9092"
export KAFKA_API_KEY="YOUR_API_KEY"
export KAFKA_API_SECRET="YOUR_API_SECRET"
```

### 4. Train the model

```bash
python train_model.py
```

This generates `data/bike_sharing.csv` (if absent), trains the classifier,
prints the accuracy and F1 report, and writes `model/bike_model.joblib`.

---

## Running the Pipeline

Open **three terminals** side by side.

### Terminal 1 — Faust Streams Processor

```bash
faust -A streams_processor worker -l info
```

Wait until you see `ready` in the Faust banner before starting the producer.

### Terminal 2 — Producer

```bash
python producer.py
```

Publishes one row per second to the `raw-data` topic.
Use `--rate 2` for faster playback or `--limit 60` to cap the run.

```
[producer]  event=     0  hr=00  temp=0.24  cnt=  13
[producer]  event=     1  hr=01  temp=0.24  cnt=   8
...
```

### Terminal 3 — Output Consumer

```bash
python consumer.py
```

Prints colour-coded predictions as they arrive (green=Low, yellow=Medium,
red=High).

```
────────────────────────────────────────────────────────────────────────
Timestamp               Event  Hour   Temp  Actual  Predicted  Confidence
────────────────────────────────────────────────────────────────────────
2025-05-18 14:03:01         0     0   0.24      13  Low          95.50%
2025-05-18 14:03:02         1     1   0.24       8  Low          97.00%
2025-05-18 14:03:03         2     5   0.27      16  Low          94.50%
2025-05-18 14:03:04         3     7   0.28     250  Medium       89.00%
2025-05-18 14:03:05         4     8   0.30     412  High         91.00%
```

---

## Component Reference

### `producer.py`

| Flag | Default | Description |
|---|---|---|
| `--rate` | `1.0` | Rows per second |
| `--limit` | `None` (full dataset) | Stop after N rows |

### `streams_processor.py`

Faust app ID: `bike-sharing-processor`  
Input topic: `raw-data` · Output topic: `predictions`  
Message schemas: `RawRecord`, `PredictionRecord` (both Faust `Record` subclasses)

### `consumer.py`

| Flag | Default | Description |
|---|---|---|
| `--no-color` | off | Disable ANSI colours (useful when logging to file) |

---

## Video Demo

> **[Link to demo video]** — replace with your YouTube unlisted / Google Drive
> link before submission.

---

## Submission Checklist

- [x] `producer.py` — streams raw-data topic at ~1 row/s
- [x] `streams_processor.py` — Faust `@app.agent` topology, no plain consumer loop
- [x] `consumer.py` — prints predictions in real time
- [x] `train_model.py` — offline training with accuracy + F1 output
- [x] `model/bike_model.joblib` — serialised model
- [x] `requirements.txt`
- [x] `README.md`
- [ ] Video demo link (add before submitting)
