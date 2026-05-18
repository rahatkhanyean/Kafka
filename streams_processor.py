"""
Faust Streams Processor — real-time Bike Sharing demand classifier.

Consumes raw sensor records from the ``raw-data`` topic, applies the
pre-trained Random Forest model, and publishes enriched prediction messages
to the ``predictions`` topic.  This component uses the Faust Streams API so
the processing loop is managed internally (not a hand-rolled consumer loop).

Usage
-----
    faust -A streams_processor worker -l info

Environment variables
---------------------
KAFKA_BOOTSTRAP_SERVERS   Broker address(es), default ``localhost:9092``
KAFKA_API_KEY             Confluent Cloud SASL username  (optional)
KAFKA_API_SECRET          Confluent Cloud SASL password  (optional)

The processor intentionally keeps ML inference synchronous inside the async
agent.  The model.predict() call is fast (< 1 ms per row) so it does not
block the event loop in any meaningful way for this demo workload.
"""

import os
import ssl
import joblib
import numpy as np
import faust

# ── load model at startup (once) ─────────────────────────────────────────────
MODEL_PATH = os.path.join("model", "bike_model.joblib")

_bundle       = joblib.load(MODEL_PATH)
_model        = _bundle["model"]
_feature_cols = _bundle["feature_cols"]
_le           = _bundle["label_encoder"]
_low_thresh   = _bundle["low_thresh"]
_high_thresh  = _bundle["high_thresh"]

print(f"[processor] Model loaded from {MODEL_PATH} "
      f"(acc={_bundle['accuracy']}, f1={_bundle['f1_macro']})")


# ── Faust app configuration ──────────────────────────────────────────────────

def _broker_url() -> str:
    """Return a Faust-style broker URL built from environment variables."""
    servers  = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    api_key  = os.environ.get("KAFKA_API_KEY", "")
    # Faust expects 'kafka://' scheme; SSL is handled via broker_credentials
    return f"kafka://{servers}"


def _credentials():
    """Return SASLCredentials for Confluent Cloud, or None for local Kafka."""
    api_key    = os.environ.get("KAFKA_API_KEY", "")
    api_secret = os.environ.get("KAFKA_API_SECRET", "")
    if not api_key:
        return None
    ctx = ssl.create_default_context()
    return faust.SASLCredentials(
        username=api_key,
        password=api_secret,
        ssl_context=ctx,
    )


_creds = _credentials()
app = faust.App(
    "bike-sharing-processor",
    broker=_broker_url(),
    broker_credentials=_creds,
    topic_partitions=1,
    # Disable automatic commit log to keep demo output clean
    stream_wait_empty=False,
)


# ── message schemas (Faust Records) ─────────────────────────────────────────

class RawRecord(faust.Record, serializer="json"):
    """Schema for messages arriving on the raw-data topic."""
    season:     int
    yr:         int
    mnth:       int
    hr:         int
    holiday:    int
    weekday:    int
    workingday: int
    weathersit: int
    temp:       float
    atemp:      float
    hum:        float
    windspeed:  float
    cnt:        int
    event_id:   int = 0


class PredictionRecord(faust.Record, serializer="json"):
    """Schema for messages published to the predictions topic."""
    event_id:        int
    hr:              int
    temp:            float
    actual_cnt:      int
    predicted_class: str   # "Low" | "Medium" | "High"
    confidence:      float
    low_thresh:      float
    high_thresh:     float


# ── topics ───────────────────────────────────────────────────────────────────

raw_topic         = app.topic("raw-data",    value_type=RawRecord)
predictions_topic = app.topic("predictions", value_type=PredictionRecord)


# ── Faust agent (stream processor) ──────────────────────────────────────────

@app.agent(raw_topic)
async def process_bike_record(records):
    """
    Core Faust agent.

    For every incoming RawRecord:
      1. Build the feature vector in the same column order as training.
      2. Run model.predict() and model.predict_proba() for confidence.
      3. Publish a PredictionRecord to the predictions topic.
    """
    async for record in records:
        # --- feature vector -------------------------------------------------
        features = np.array([[
            record.season,
            record.yr,
            record.mnth,
            record.hr,
            record.holiday,
            record.weekday,
            record.workingday,
            record.weathersit,
            record.temp,
            record.atemp,
            record.hum,
            record.windspeed,
        ]])

        # --- inference ------------------------------------------------------
        pred_idx    = int(_model.predict(features)[0])
        proba       = _model.predict_proba(features)[0]
        label       = _le.inverse_transform([pred_idx])[0]
        confidence  = float(proba[pred_idx])

        # --- publish result -------------------------------------------------
        prediction = PredictionRecord(
            event_id        = record.event_id,
            hr              = record.hr,
            temp            = record.temp,
            actual_cnt      = record.cnt,
            predicted_class = label,
            confidence      = round(confidence, 4),
            low_thresh      = _low_thresh,
            high_thresh     = _high_thresh,
        )

        await predictions_topic.send(value=prediction)

        print(
            f"[processor] event={record.event_id:>6}  "
            f"cnt={record.cnt:>4}  →  {label:<6}  "
            f"(conf={confidence:.2f})"
        )
