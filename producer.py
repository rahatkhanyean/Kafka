"""
Kafka Producer — Bike Sharing live event streamer.

Reads rows from data/bike_sharing.csv (or generates them on-the-fly) and
publishes each row as a JSON message to the ``raw-data`` Kafka topic at
approximately one message per second, simulating a live sensor feed.

Usage
-----
    python producer.py [--rate ROWS_PER_SECOND] [--limit MAX_ROWS]

Environment variables
---------------------
KAFKA_BOOTSTRAP_SERVERS   Broker address(es), default ``localhost:9092``
KAFKA_API_KEY             Confluent Cloud SASL username  (optional)
KAFKA_API_SECRET          Confluent Cloud SASL password  (optional)

If KAFKA_API_KEY is set the producer connects with SASL_SSL; otherwise it
uses a plain (no-auth) PLAINTEXT connection suitable for local Kafka.
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError

# ── defaults ────────────────────────────────────────────────────────────────
TOPIC               = "raw-data"
DEFAULT_RATE        = 1.0          # rows per second
DATA_PATH           = os.path.join("data", "bike_sharing.csv")


# ── helpers ─────────────────────────────────────────────────────────────────

def build_producer() -> KafkaProducer:
    """Create and return a KafkaProducer configured from environment variables."""
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    api_key   = os.environ.get("KAFKA_API_KEY", "")
    api_secret= os.environ.get("KAFKA_API_SECRET", "")

    common = dict(
        bootstrap_servers=bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        acks="all",
        retries=3,
    )

    if api_key:
        # Confluent Cloud — SASL/SSL
        return KafkaProducer(
            **common,
            security_protocol="SASL_SSL",
            sasl_mechanism="PLAIN",
            sasl_plain_username=api_key,
            sasl_plain_password=api_secret,
        )

    # Local Kafka
    return KafkaProducer(**common)


def load_dataset() -> pd.DataFrame:
    """Load the Bike Sharing CSV; auto-generate it if not present."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)

    print(f"[producer] {DATA_PATH} not found — generating synthetic data ...")
    sys.path.insert(0, os.path.dirname(__file__))
    from data.generate_data import generate_bike_sharing
    os.makedirs("data", exist_ok=True)
    df = generate_bike_sharing()
    df.to_csv(DATA_PATH, index=False)
    return df


def on_send_error(exc: Exception) -> None:
    """Log delivery errors without crashing the producer loop."""
    print(f"[producer] ERROR — message not delivered: {exc}")


# ── main loop ────────────────────────────────────────────────────────────────

def run(rate: float = DEFAULT_RATE, limit: int | None = None) -> None:
    """Stream dataset rows to Kafka at *rate* rows per second."""
    df       = load_dataset()
    producer = build_producer()
    delay    = 1.0 / rate
    total    = len(df) if limit is None else min(limit, len(df))

    print(f"[producer] Connected → topic='{TOPIC}'")
    print(f"[producer] Streaming {total:,} rows at {rate} row/s …\n")

    for i, (_, row) in enumerate(df.head(total).iterrows()):
        record = row.to_dict()
        # Attach a sequence number for traceability
        record["event_id"] = i

        producer.send(TOPIC, value=record).add_errback(on_send_error)

        # Print a summary every row so the terminal stays readable
        print(
            f"[producer]  event={i:>6}  hr={int(record['hr']):02d}  "
            f"temp={record['temp']:.2f}  cnt={int(record['cnt']):>4}"
        )

        time.sleep(delay)

    producer.flush()
    print("\n[producer] Done — all rows published.")


# ── entry point ──────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bike Sharing Kafka Producer")
    p.add_argument(
        "--rate", type=float, default=DEFAULT_RATE,
        help="Rows to publish per second (default: 1.0)"
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Max rows to publish (default: entire dataset)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        run(rate=args.rate, limit=args.limit)
    except KeyboardInterrupt:
        print("\n[producer] Interrupted by user.")
