"""FastAPI wrapper for the fraud triage pipeline."""
import json
import logging
import time
from collections import defaultdict
from threading import Lock

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from prepare import add_features
from train import build_final_model
from casenote import run_case

logging.basicConfig(filename="service.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fraud_triage")
app = FastAPI(title="Fraud Triage Service")
MODEL_BUNDLE = build_final_model()
MODEL = MODEL_BUNDLE["model"]
COLUMNS = MODEL_BUNDLE["columns"]
MEDIAN = MODEL_BUNDLE["median"]
THRESHOLD = MODEL_BUNDLE["threshold"]
CATEGORIES = {"Grocery", "Electronics", "Travel", "Food Delivery", "Fashion", "Utilities"}
COUNTERS = {"requests_total": 0, "success_total": 0, "error_total": 0,
            "human_review_total": 0, "latency_total_ms": 0.0}
COUNTER_LOCK = Lock()


class Txn(BaseModel):
    txn_id: str
    merchant_category: str
    city: str
    payment_mode: str
    amount_inr: float
    hour_of_day: int
    txn_count_last_24h: int
    avg_amount_30d_inr: float
    distance_from_home_km: float | None = None
    is_new_device: int
    account_age_days: int


class CaseIn(Txn):
    question: str


@app.middleware("http")
async def observe(request: Request, call_next):
    t0 = time.perf_counter()
    request.state.info = {}
    try:
        response = await call_next(request)
    except Exception:
        # Unexpected exceptions are logged as one request line, without a traceback in the log.
        response = JSONResponse(status_code=500, content={"error": "internal service error"})
    if request.method == "POST" and request.url.path in {"/score", "/case"}:
        ms = (time.perf_counter() - t0) * 1000
        status = response.status_code
        info = request.state.info
        human_review = bool(info.get("human_review", False)) if status < 400 else False
        if status >= 500:
            level = logging.ERROR
        elif status >= 400 or info.get("safe_default") or info.get("guardrail_override") or info.get("large_payment_hold"):
            level = logging.WARNING
        else:
            level = logging.INFO
        endpoint = request.url.path
        message = f"endpoint={endpoint} status_code={status} latency_ms={ms:.2f}"
        if info.get("txn_id"):
            message += f" txn_id={info['txn_id']}"
        if info.get("fraud_probability") is not None:
            message += f" fraud_probability={info['fraud_probability']:.4f}"
        logger.log(level, message)
        with COUNTER_LOCK:
            COUNTERS["requests_total"] += 1
            COUNTERS["success_total"] += int(status < 400)
            COUNTERS["error_total"] += int(status >= 400)
            COUNTERS["human_review_total"] += int(human_review)
            COUNTERS["latency_total_ms"] += ms
    return response


def validate_business_values(t: Txn):
    checks = [
        (not 1 <= t.amount_inr <= 1_000_000, "amount_inr", "amount_inr must be between 1 and 1000000"),
        (not 0 <= t.hour_of_day <= 23, "hour_of_day", "hour_of_day must be between 0 and 23"),
        (t.txn_count_last_24h < 0, "txn_count_last_24h", "txn_count_last_24h must be non-negative"),
        (t.avg_amount_30d_inr <= 0, "avg_amount_30d_inr", "avg_amount_30d_inr must be greater than 0"),
        (t.merchant_category.strip().title() not in CATEGORIES, "merchant_category", "unknown merchant_category"),
    ]
    for failed, field, message in checks:
        if failed:
            return {"error": message, "field": field}
    return None


def score_transaction(t: Txn):
    row = {"amount_inr": t.amount_inr, "hour_of_day": t.hour_of_day,
           "txn_count_last_24h": t.txn_count_last_24h, "avg_amount_30d_inr": t.avg_amount_30d_inr,
           "distance_from_home_km": t.distance_from_home_km, "is_new_device": t.is_new_device,
           "account_age_days": t.account_age_days,
           "merchant_category": t.merchant_category.strip().title(),
           "city": t.city.strip().title(), "payment_mode": t.payment_mode}
    df = add_features(pd.DataFrame([row]))
    df["distance_from_home_km"] = df["distance_from_home_km"].fillna(MEDIAN)
    encoded = pd.get_dummies(df, columns=["merchant_category", "city", "payment_mode"], dtype=int)
    encoded = encoded.reindex(columns=COLUMNS, fill_value=0)
    prob = float(MODEL.predict_proba(encoded)[0, 1])
    flagged = prob >= THRESHOLD
    decision = "hold_for_review" if flagged else "approve"
    large_payment_hold = bool(not flagged and t.amount_inr >= 100_000 and t.is_new_device == 1)
    if large_payment_hold:
        decision = "hold_for_review"
    return prob, flagged, decision, large_payment_hold


def base_response(t: Txn, prob, flagged, decision, large_hold):
    return {"txn_id": t.txn_id, "fraud_probability": round(prob, 4), "flagged": bool(flagged),
            "decision": decision, "human_review": decision == "hold_for_review" or decision == "block_and_notify",
            "large_payment_hold": large_hold}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    with COUNTER_LOCK:
        total = COUNTERS["requests_total"]
        success = COUNTERS["success_total"]
        errors = COUNTERS["error_total"]
        reviews = COUNTERS["human_review_total"]
        latency = COUNTERS["latency_total_ms"]
    return {"requests_total": total, "success_total": success, "error_total": errors,
            "success_rate": round(success / total, 4) if total else 0.0,
            "error_rate": round(errors / total, 4) if total else 0.0,
            "avg_latency_ms": round(latency / total, 4) if total else 0.0,
            "human_review_total": reviews,
            "human_review_rate": round(reviews / total, 4) if total else 0.0}


@app.post("/score")
def score(t: Txn, request: Request):
    request.state.info["txn_id"] = t.txn_id
    bad = validate_business_values(t)
    if bad:
        return JSONResponse(status_code=400, content={"error": bad["error"], "field": bad["field"]})
    prob, flagged, decision, large_hold = score_transaction(t)
    result = base_response(t, prob, flagged, decision, large_hold)
    request.state.info.update({"txn_id": t.txn_id, "fraud_probability": prob,
                               "human_review": result["human_review"], "large_payment_hold": large_hold})
    result.pop("large_payment_hold", None)
    return result


@app.post("/case")
def case(t: CaseIn, request: Request):
    request.state.info["txn_id"] = t.txn_id
    bad = validate_business_values(t)
    if bad:
        return JSONResponse(status_code=400, content={"error": bad["error"], "field": bad["field"]})
    prob, flagged, decision, large_hold = score_transaction(t)
    request.state.info.update({"txn_id": t.txn_id, "fraud_probability": prob,
                               "large_payment_hold": large_hold})
    result = base_response(t, prob, flagged, decision, large_hold)
    result.update({"case_note": None, "attempts": 0, "safe_default": False, "guardrail_override": False})
    if flagged:
        try:
            outcome = run_case(t.txn_id, prob, t.amount_inr, t.question)
        except ConnectionError:
            return JSONResponse(status_code=503, content={"error": "case-note service temporarily unavailable, try again later"})
        result["case_note"] = outcome["case_note"]
        result["attempts"] = outcome["attempts"]
        result["safe_default"] = outcome["safe_default"]
        result["guardrail_override"] = outcome["guardrail_override"]
        if outcome["status"] == "no_policy_match":
            decision = "hold_for_review"
        elif outcome["case_note"]:
            action = outcome["case_note"]["recommended_action"]
            # Flagged transactions must never be auto-approved, including probabilities < 0.5.
            if action == "approve":
                outcome["case_note"]["recommended_action"] = "hold_for_review"
                result["guardrail_override"] = True
                action = "hold_for_review"
            decision = action
        result["human_review"] = decision in {"hold_for_review", "block_and_notify"} or result["safe_default"]
    else:
        result["human_review"] = decision == "hold_for_review" or decision == "block_and_notify"
    result["decision"] = decision
    request.state.info.update({"txn_id": t.txn_id, "fraud_probability": prob,
                               "human_review": result["human_review"], "safe_default": result["safe_default"],
                               "guardrail_override": result["guardrail_override"], "large_payment_hold": large_hold})
    result.pop("large_payment_hold", None)
    return result
