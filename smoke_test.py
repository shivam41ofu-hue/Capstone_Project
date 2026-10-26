"""Run the 11 prescribed smoke checks against a running local service."""
import time
import requests

BASE = "http://127.0.0.1:8000"
BENIGN = {"merchant_category": "Grocery", "city": "Pune", "payment_mode": "UPI", "amount_inr": 1500,
          "hour_of_day": 13, "txn_count_last_24h": 0, "avg_amount_30d_inr": 1600,
          "distance_from_home_km": 3.0, "is_new_device": 0, "account_age_days": 900}
RISKY = {"merchant_category": "Electronics", "city": "Mumbai", "payment_mode": "Card", "amount_inr": 45000,
         "hour_of_day": 2, "txn_count_last_24h": 9, "avg_amount_30d_inr": 3000,
         "distance_from_home_km": 60.0, "is_new_device": 1, "account_age_days": 40}
BIG = {**BENIGN, "merchant_category": "Utilities", "payment_mode": "NetBanking", "amount_inr": 150000,
       "avg_amount_30d_inr": 250000, "distance_from_home_km": 0.0, "is_new_device": 1, "account_age_days": 2200}


def wait_for_service():
    end = time.time() + 120
    while time.time() < end:
        try:
            r = requests.get(BASE + "/health", timeout=2)
            if r.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("service did not start within 120 seconds")


def main():
    wait_for_service()
    checks = []
    def add(qid, category, method, path, body, expected, check):
        try:
            r = requests.request(method, BASE + path, json=body, timeout=30)
            data = r.json()
            passed = r.status_code == expected and check(data)
            checks.append({"id": qid, "category": category, "expected": expected,
                           "observed": r.status_code, "result": "PASS" if passed else "FAIL"})
        except Exception as exc:
            checks.append({"id": qid, "category": category, "expected": expected,
                           "observed": f"exception: {exc}", "result": "FAIL"})
    add("Q1", "happy", "GET", "/health", None, 200, lambda d: d.get("status") == "ok")
    add("Q2", "happy", "POST", "/score", {**BENIGN, "txn_id": "T800001"}, 200,
        lambda d: d.get("flagged") is False and d.get("decision") == "approve" and d.get("human_review") is False)
    add("Q3", "escalation", "POST", "/score", {**RISKY, "txn_id": "T800002"}, 200,
        lambda d: d.get("flagged") is True and d.get("decision") == "hold_for_review" and d.get("human_review") is True)
    add("Q4", "escalation", "POST", "/score", {**BIG, "txn_id": "T800003"}, 200,
        lambda d: d.get("flagged") is False and d.get("decision") == "hold_for_review" and d.get("human_review") is True)
    add("Q5", "edge", "POST", "/score", {**BENIGN, "txn_id": "T800004", "amount_inr": -5}, 400,
        lambda d: "error" in d and d.get("field") == "amount_inr")
    missing_hour = {**BENIGN, "txn_id": "T800005"}; missing_hour.pop("hour_of_day")
    add("Q6", "edge", "POST", "/score", missing_hour, 422, lambda d: True)
    add("Q7", "escalation", "POST", "/case", {**RISKY, "txn_id": "T900002", "question": "more than 6 UPI payments in 24 hours velocity alert"}, 200,
        lambda d: d.get("flagged") is True and d.get("attempts") == 2 and d.get("safe_default") is False and (d.get("case_note") or {}).get("policy_id") == "P2" and d.get("human_review") is True)
    add("Q8", "escalation", "POST", "/case", {**RISKY, "txn_id": "T900005", "question": "electronics merchant payment on an account younger than 30 days"}, 200,
        lambda d: d.get("attempts") == 3 and d.get("safe_default") is True and d.get("decision") == "hold_for_review" and d.get("human_review") is True)
    add("Q9", "edge", "POST", "/case", {**RISKY, "txn_id": "T900099", "question": "card payment above INR 50000 from a new device"}, 503,
        lambda d: "error" in d)
    add("Q10", "happy", "POST", "/case", {**BENIGN, "txn_id": "T800001", "question": "card payment above INR 50000 from a new device"}, 200,
        lambda d: d.get("case_note") is None and d.get("attempts") == 0 and d.get("decision") == "approve" and d.get("human_review") is False)
    add("Q11", "edge", "POST", "/case", {**RISKY, "txn_id": "T800006", "question": "what is the cafeteria menu today"}, 200,
        lambda d: d.get("case_note") is None and d.get("attempts") == 0 and d.get("decision") == "hold_for_review" and d.get("human_review") is True)
    print("ID  CATEGORY    EXPECTED  OBSERVED  RESULT")
    for row in checks:
        print(f"{row['id']:<3} {row['category']:<11} {row['expected']:<8} {str(row['observed']):<9} {row['result']}")
    print("\nCATEGORY PASS COUNTS")
    for cat in ["happy", "escalation", "edge"]:
        group = [r for r in checks if r["category"] == cat]
        print(f"{cat}: {sum(r['result']=='PASS' for r in group)}/{len(group)}")
    print("\nMETRICS")
    print(requests.get(BASE + "/metrics", timeout=10).json())
    if any(r["result"] != "PASS" for r in checks):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
