"""TF-IDF policy retrieval with a calibrated refusal threshold."""
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

POLICY_PATH = Path("fraud_policies.json")
QUERY_PATH = Path("retrieval_queries.json")
POLICIES = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
VECTORIZER = TfidfVectorizer()
_POLICY_MATRIX = VECTORIZER.fit_transform([p["text"] for p in POLICIES])
MIN_SCORE = 0.45  # above highest out-of-scope score (0.4251), <= lowest in-scope (0.4862)


def retrieve(query: str):
    scores = cosine_similarity(VECTORIZER.transform([query]), _POLICY_MATRIX)[0]
    i = int(scores.argmax())
    return POLICIES[i]["policy_id"], POLICIES[i]["text"], float(scores[i])


def main():
    queries = json.loads(QUERY_PATH.read_text(encoding="utf-8"))
    rows = []
    hits = 0
    in_scope_scores, out_scope_scores = [], []
    for item in queries:
        pid, _, score = retrieve(item["query"])
        expected = item["expected_policy"]
        if expected is not None:
            hits += pid == expected
            in_scope_scores.append(score)
        else:
            out_scope_scores.append(score)
        rows.append({"query": item["query"], "expected": expected, "retrieved": pid, "score": score,
                     "accepted": score >= MIN_SCORE})
    print("RETRIEVAL TABLE")
    print(json.dumps(rows, indent=2))
    print(f"in_scope_top1_hits={hits}/8")
    print(f"highest_out_of_scope_score={max(out_scope_scores):.4f}; lowest_in_scope_score={min(in_scope_scores):.4f}; MIN_SCORE={MIN_SCORE:.4f}")

if __name__ == "__main__":
    main()
