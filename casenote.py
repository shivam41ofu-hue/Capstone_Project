"""Policy-grounded case note generation with schema-driven validation."""
import json
from pathlib import Path
from retrieval import retrieve, MIN_SCORE
from mock_llm import generate, reset

CASE_PATH = Path("case_scenarios.json")
CASE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        "recommended_team": {"type": "string", "enum": ["card_risk", "upi_risk", "merchant_ops", "customer_care"]},
        "recommended_action": {"type": "string", "enum": ["approve", "hold_for_review", "block_and_notify"]},
        "policy_id": {"type": "string", "enum": ["P1", "P2", "P3", "P4", "P5", "P6"]},
    },
    "required": ["summary", "priority", "recommended_team", "recommended_action", "policy_id"],
}


def build_prompt(txn_id, fraud_probability, amount_inr, policy_id, policy_text):
    return f"""ROLE: You are a careful payment-risk analyst.
TASK: Create a concise case note for the transaction using the supplied policy.
CONSTRAINTS: Use only the policy text below. Do not invent facts. Reply with one JSON object and nothing else.
OUTPUT FORMAT: JSON object with keys summary (string), priority (low|medium|high), recommended_team (card_risk|upi_risk|merchant_ops|customer_care), recommended_action (approve|hold_for_review|block_and_notify), policy_id (P1|P2|P3|P4|P5|P6).
POLICY TEXT: {policy_text}
TXN_ID: {txn_id}
FRAUD_PROBABILITY: {fraud_probability}
AMOUNT_INR: {amount_inr}
POLICY_ID: {policy_id}"""


def validate(text, retrieved_policy_id):
    try:
        note = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, 1
    if not isinstance(note, dict):
        return None, 1
    required = CASE_SCHEMA["required"]
    props = CASE_SCHEMA["properties"]
    if any(k not in note or not isinstance(note[k], str) for k in required):
        return note, 2
    for key in required:
        if "enum" in props[key] and note[key] not in props[key]["enum"]:
            return note, 3
    if note["policy_id"] != retrieved_policy_id:
        return note, 3
    return note, 0


def run_case(txn_id, fraud_probability, amount_inr, question, max_attempts=3):
    policy_id, policy_text, score = retrieve(question)
    if score < MIN_SCORE:
        return {"status": "no_policy_match", "attempts": 0, "failures": [], "case_note": None,
                "safe_default": False, "guardrail_override": False, "retrieved_policy_id": policy_id,
                "retrieval_score": score}
    prompt = build_prompt(txn_id, fraud_probability, amount_inr, policy_id, policy_text)
    failures = []
    note = None
    attempts = 0
    for _ in range(max_attempts):
        attempts += 1
        # ConnectionError intentionally propagates to the API layer.
        raw = generate(prompt)
        note, gate = validate(raw, policy_id)
        if gate == 0:
            break
        failures.append(gate)
        note = None
    safe_default = note is None
    if safe_default:
        note = {"summary": "Automated case note unavailable; manual review required.",
                "priority": "high", "recommended_team": "customer_care",
                "recommended_action": "hold_for_review", "policy_id": policy_id}
    override = False
    if note["recommended_action"] == "approve" and float(fraud_probability) >= 0.5:
        note["recommended_action"] = "hold_for_review"
        override = True
    return {"status": "ok", "attempts": attempts, "failures": failures, "case_note": note,
            "safe_default": safe_default, "guardrail_override": override,
            "retrieved_policy_id": policy_id, "retrieval_score": score}


def main():
    reset()
    scenarios = json.loads(CASE_PATH.read_text(encoding="utf-8"))
    rows = []
    for i, s in enumerate(scenarios, 1):
        r = run_case(s["txn_id"], s["fraud_probability"], s["amount_inr"], s["question"])
        rows.append({"scenario": f"S{i} ({s['txn_id']})", "retrieved_policy": r.get("retrieved_policy_id"),
                     "attempts": r["attempts"], "failures": r["failures"], "status": r["status"],
                     "safe_default": "yes" if r["safe_default"] else "no",
                     "final_action": (r["case_note"] or {}).get("recommended_action"),
                     "override": "yes" if r["guardrail_override"] else "no"})
    print("CASE SCENARIOS")
    print(json.dumps(rows, indent=2))
    print(f"total_model_calls={sum(r['attempts'] for r in rows)}")
    print(f"first_call_successes={sum(r['attempts']==1 and r['safe_default']=='no' for r in rows)}")
    print(f"safe_defaults={sum(r['safe_default']=='yes' for r in rows)}; guardrail_overrides={sum(r['override']=='yes' for r in rows)}")

if __name__ == "__main__":
    main()
