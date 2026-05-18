"""
Kafka Output Consumer — prints live demand predictions to the terminal.

Reads ``PredictionRecord`` messages from the ``predictions`` topic and formats
each one as a human-readable line, optionally colour-coded by demand class.

Usage
-----
    python consumer.py [--no-color]

Environment variables
---------------------
KAFKA_BOOTSTRAP_SERVERS   Broker address(es), default ``localhost:9092``
KAFKA_API_KEY             Confluent Cloud SASL username  (optional)
KAFKA_API_SECRET          Confluent Cloud SASL password  (optional)
"""

import argparse
import json
import os
import sys
from datetime import datetime

from kafka import KafkaConsumer
from kafka.errors import KafkaError

# ── topic & consumer group ───────────────────────────────────────────────────
TOPIC          = "predictions"
CONSUMER_GROUP = "bike-sharing-output-consumer"


# ── ANSI colour helpers ───────────────────────────────────────────────────────
_RESET  = "\033[0m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"

_CLASS_COLOR = {
    "Low":    _GREEN,
    "Medium": _YELLOW,
    "High":   _RED,
}


def _colorize(text: str, demand_class: str, use_color: bool) -> str:
    if not use_color:
        return text
    color = _CLASS_COLOR.get(demand_class, "")
    return f"{color}{text}{_RESET}"


# ── consumer factory ─────────────────────────────────────────────────────────

def build_consumer() -> KafkaConsumer:
    """Create and return a KafkaConsumer configured from environment variables."""
    bootstrap  = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    api_key    = os.environ.get("KAFKA_API_KEY", "")
    api_secret = os.environ.get("KAFKA_API_SECRET", "")

    common = dict(
        bootstrap_servers=bootstrap,
        group_id=CONSUMER_GROUP,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
    )

    if api_key:
        return KafkaConsumer(
            TOPIC,
            **common,
            security_protocol="SASL_SSL",
            sasl_mechanism="PLAIN",
            sasl_plain_username=api_key,
            sasl_plain_password=api_secret,
        )

    return KafkaConsumer(TOPIC, **common)


# ── display logic ─────────────────────────────────────────────────────────────

_HEADER_PRINTED = False

def _print_header() -> None:
    """Print the column header once when the first message arrives."""
    global _HEADER_PRINTED
    if _HEADER_PRINTED:
        return
    _HEADER_PRINTED = True
    print(
        f"\n{'─'*72}\n"
        f"{'Timestamp':<22}  {'Event':>7}  {'Hour':>4}  "
        f"{'Temp':>5}  {'Actual':>6}  {'Predicted':<9}  {'Confidence':>10}\n"
        f"{'─'*72}"
    )


def _format_row(msg: dict, use_color: bool) -> str:
    """Return a formatted display row for one prediction message."""
    ts         = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    event_id   = msg.get("event_id", "?")
    hr         = msg.get("hr", "?")
    temp       = msg.get("temp", 0.0)
    actual     = msg.get("actual_cnt", "?")
    pred_class = msg.get("predicted_class", "?")
    confidence = msg.get("confidence", 0.0)

    row = (
        f"{ts:<22}  {str(event_id):>7}  {str(hr):>4}  "
        f"{temp:>5.2f}  {str(actual):>6}  {pred_class:<9}  "
        f"{confidence:>9.2%}"
    )
    return _colorize(row, pred_class, use_color)


# ── main loop ────────────────────────────────────────────────────────────────

def run(use_color: bool = True) -> None:
    """Poll the predictions topic and pretty-print every arriving message."""
    consumer = build_consumer()
    print(f"[consumer] Listening on topic='{TOPIC}'  (Ctrl-C to stop) …")

    try:
        for message in consumer:
            msg = message.value
            _print_header()
            print(_format_row(msg, use_color))
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n[consumer] Stopped by user.")
    finally:
        consumer.close()


# ── entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bike Sharing Predictions Consumer")
    p.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI colour output (useful when piping to a file)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(use_color=not args.no_color)
