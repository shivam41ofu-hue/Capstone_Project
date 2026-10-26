# Razorpay Merchant Risk Desk — Fraud Triage Service

## Overview
This repository implements a local fraud-triage service over a fictional, synthetic payment dataset.
Part A audits, cleans, and engineers payment features without mutating its input.
Part B compares imbalance treatments, tunes a random forest, and chooses an alert threshold from validation recall.
Part C retrieves a fraud policy and generates validated JSON case notes through the supplied offline mock language model.
Part D exposes the pipeline through FastAPI with input and output guardrails, one-line request logging, and in-memory metrics.
All amounts are INR; the scenario and records are fictional and are not real Razorpay data or policy.
No paid API, cloud deployment, or network-based language model is required.

## Reproduce
Run these commands from the repository root. Python 3.11 or newer is required.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python generate_data.py
python prepare.py
python train.py
python retrieval.py
python casenote.py
```

For the Part D smoke test, use a second terminal from the same repository root. Start the service, wait for `Application startup complete`, then run the smoke test from the first terminal. Restart the server and remove the old log before every final smoke run because the mock language model keeps per-process call counts and the log appends.

```bash
# Terminal 1
rm -f service.log
uvicorn app:app --port 8000

# Terminal 2 (same virtual environment and repository)
python smoke_test.py
```

On Windows PowerShell, use `Remove-Item service.log -ErrorAction SilentlyContinue` in place of `rm -f service.log`. Stop the server with Ctrl+C after the smoke test. The generated data files are committed for convenience; `python generate_data.py` deterministically recreates them. `generate_data.py` and `mock_llm.py` are supplied files and must remain unmodified.

## Part A results

### Audit

| Check | Result |
|---|---:|
| Raw rows × columns | 6,090 × 14 |
| Exact duplicate rows | 90 |
| Missing `amount_inr` after de-duplication | 76 |
| Missing `distance_from_home_km` after de-duplication | 193 |
| Amount below INR 1 or above INR 1,000,000 after de-duplication | 5 |
| Distinct raw spellings of `merchant_category` | 24 |
| Distinct raw spellings of `city` | 24 |

### Cleaned-data checkpoints

| Check | Result |
|---|---:|
| Clean rows | 5,919 |
| `merchant_category` counts | Grocery 1,406; Food Delivery 1,326; Fashion 976; Electronics 845; Utilities 685; Travel 681 |
| `city` counts | Bengaluru 1,033; Chennai 1,025; Kolkata 996; Delhi 979; Pune 944; Mumbai 942 |
| `payment_mode` counts | UPI 3,221; Card 1,450; NetBanking 731; Wallet 517 |
| Minimum / maximum amount | INR 141.37 / INR 105,883.95 |
| Remaining missing distance values | 191 |
| `is_night` sum / `high_value` sum | 365 / 17 |
| Maximum `amount_vs_avg_ratio` | 13.11 |

The distance values are not imputed in Part A because the imputation median must be learned from training rows only, avoiding information leakage from validation and test rows.

### Leakage audit and class balance

| `chargeback_filed` | Rows | Fraud rows |
|---:|---:|---:|
| 0 | 5,546 | 62 |
| 1 | 373 | 304 |

The cleaned table has 366 fraud rows out of 5,919 (6.18%). A “never fraud” rule has 93.82% accuracy and 0 recall. Accuracy is misleading because the majority class dominates the dataset, so a model can score highly while missing every fraud. Precision, recall, F1 and the confusion matrix make the false-positive/false-negative trade-off visible.

`chargeback_filed` is known after settlement and therefore leaks future outcome information; it is excluded from model inputs. `txn_id` and `customer_id` are identifiers rather than generalizable transaction-risk signals and can encourage memorization. The excluded columns are `chargeback_filed`, `txn_id`, `customer_id` and `is_fraud`. Before one-hot encoding, model inputs are `amount_inr`, `hour_of_day`, `txn_count_last_24h`, `avg_amount_30d_inr`, `distance_from_home_km`, `is_new_device`, `account_age_days`, `amount_vs_avg_ratio`, `is_night`, `high_value`, `merchant_category`, `city` and `payment_mode`.

## Part B results

### Split and feature columns

| Split | Rows | Fraud rows |
|---|---:|---:|
| Train | 3,551 | 220 |
| Validation | 1,184 | 73 |
| Test | 1,184 | 73 |

The training-only median for `distance_from_home_km` is **12.6 km**. Missing distance values in all three splits are filled with that value. The exact 26 model columns, in order, are:

```text
amount_inr
hour_of_day
txn_count_last_24h
avg_amount_30d_inr
distance_from_home_km
is_new_device
account_age_days
amount_vs_avg_ratio
is_night
high_value
merchant_category_Electronics
merchant_category_Fashion
merchant_category_Food Delivery
merchant_category_Grocery
merchant_category_Travel
merchant_category_Utilities
city_Bengaluru
city_Chennai
city_Delhi
city_Kolkata
city_Mumbai
city_Pune
payment_mode_Card
payment_mode_NetBanking
payment_mode_UPI
payment_mode_Wallet
```

### Imbalance approaches (validation, threshold 0.5)

| Model | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Plain logistic regression | 0.625 | 0.342 | 0.442 | 1,096 | 15 | 48 | 25 |
| Balanced logistic regression | 0.252 | 0.781 | 0.381 | 942 | 169 | 16 | 57 |
| Random over-sampling + logistic regression | 0.257 | 0.781 | 0.386 | 946 | 165 | 16 | 57 |

The over-sampled training table has 6,662 rows: 3,331 non-fraud and 3,331 fraud rows. A risk desk that prioritizes not missing fraud would prefer a balanced or over-sampled model because each catches 32 more validation frauds than the plain model. The cost is substantially more legitimate payments sent for review, lowering precision and increasing analyst workload.

### Cross-validation and tuning

| Model | Mean CV F1 | Standard deviation |
|---|---:|---:|
| Balanced logistic regression | 0.402 | 0.015 |
| Untuned random forest | 0.168 | 0.048 |
| Tuned random forest (best grid result) | 0.511 | 0.080 |

The grid search tested 12 combinations across five stratified folds, using only the training split. The tuned forest CV F1 standard deviation was 0.080. The best parameters in this run were `max_depth=None` and `min_samples_leaf=5`, with `n_estimators=200`, `class_weight="balanced"` and `random_state=42` fixed. The tuned forest is selected because it had the strongest mean CV F1 in this run and the feature-importance ranking gives a useful view of the signals it uses. Its fold-to-fold stability should be considered alongside the mean, and the forest is less straightforward to explain than logistic regression because it combines many decision trees. The trade-off is reduced coefficient-level interpretability compared with logistic regression.

### Validation threshold table

| Threshold | Precision | Recall | F1 | Payments flagged |
|---:|---:|---:|---:|---:|
| 0.1 | 0.140 | 0.945 | 0.244 | 492 |
| 0.2 | 0.203 | 0.836 | 0.326 | 301 |
| 0.3 | 0.290 | 0.699 | 0.410 | 176 |
| 0.4 | 0.454 | 0.603 | 0.518 | 97 |
| 0.5 | 0.518 | 0.397 | 0.450 | 56 |
| 0.6 | 0.606 | 0.274 | 0.377 | 33 |
| 0.7 | 0.700 | 0.192 | 0.301 | 20 |
| 0.8 | 0.750 | 0.082 | 0.148 | 8 |
| 0.9 | 0.000 | 0.000 | 0.000 | 0 |

`t_star` is **0.3**, the highest grid threshold whose validation recall is at least 0.65 in this run (recall 0.699 at 0.3; recall 0.603 at 0.4). The final model is evaluated once on the test split at that threshold: precision 0.328, recall 0.808, F1 0.466, with TN 990, FP 121, FN 14 and TP 59; 180 payments are flagged. This corresponds to 121 / 59 = **2.05 legitimate payments held per fraud caught**, and 14 frauds missed. A stricter threshold reduces the number of held payments but also catches fewer frauds: on validation, moving from 0.3 to 0.4 reduces flags from 176 to 97 and recall from 0.699 to 0.603. The committee should treat this as a trade-off rather than a universally optimal threshold.

### Leakage experiment and feature importance

At threshold 0.5 on validation, the tuned clean model has F1 0.450 and the model with `chargeback_filed` added as the 27th column has F1 0.822. The higher leaky score is not a genuine improvement because chargeback status is known after settlement and would not be available at the intended decision time. It gives the model information about a later outcome and therefore cannot support a fair estimate of live fraud detection.

| Feature | Importance |
|---|---:|
| `amount_vs_avg_ratio` | 0.2614 |
| `txn_count_last_24h` | 0.1634 |
| `amount_inr` | 0.1359 |
| `account_age_days` | 0.0845 |
| `distance_from_home_km` | 0.0748 |

The top two signals align with unusual spending relative to a customer's normal amount and unusually frequent payment activity. Both can be useful risk indicators, but neither proves that an individual transaction is fraudulent.

## Part C results

The TF-IDF retriever gets **8/8 in-scope queries correct**. The highest out-of-scope score is 0.4251 and the lowest in-scope score is 0.4862; `MIN_SCORE=0.4500` sits between them. The two out-of-scope questions are refused without a model call.

| Query | Expected policy | Retrieved policy | Cosine score |
|---|---|---|---:|
| card payment above INR 50000 from a new device | P1 | P1 | 0.5501 |
| how many UPI payments in 24 hours trigger a velocity alert | P2 | P2 | 0.5622 |
| payment at 3 AM far away from the home city | P3 | P3 | 0.4948 |
| customer reports unauthorised payment and chargeback case | P4 | P4 | 0.5689 |
| electronics merchant payment on an account younger than 30 days | P5 | P5 | 0.5641 |
| which payments are approved automatically without manual review | P6 | P6 | 0.6250 |
| temporary hold on repeated UPI payments within a day | P2 | P2 | 0.4862 |
| what happens after a customer reports an unauthorised payment | P4 | P4 | 0.4867 |
| what is the cafeteria menu today | null | P3 (refused) | 0.4251 |
| how do I reset my email password | null | P1 (refused) | 0.0000 |

| Scenario | Policy | Attempts | Failed gates | Status | Safe default | Final action | Override |
|---|---|---:|---|---|---|---|---|
| S1 (T900001) | P1 | 1 | [] | ok | no | hold_for_review | no |
| S2 (T900002) | P2 | 2 | [1] | ok | no | hold_for_review | no |
| S3 (T900003) | P3 | 2 | [2] | ok | no | hold_for_review | no |
| S4 (T900004) | P4 | 2 | [3] | ok | no | block_and_notify | no |
| S5 (T900005) | P5 | 3 | [1, 1, 1] | ok | yes | hold_for_review | no |
| S6 (T900006) | P2 | 1 | [] | ok | no | hold_for_review | yes |
| S7 (T900007) | P1 | 2 | [3] | ok | no | hold_for_review | no |
| S8 (T900008) | P4 | 1 | [] | ok | no | block_and_notify | no |
| S9 (T900009) | P3 | 3 | [1, 2] | ok | no | hold_for_review | no |
| S10 (T900010) | P2 | 2 | [3] | ok | no | hold_for_review | no |

Totals: **19 model calls**, 3 valid first-call notes (S1, S6, S8), 6 scenarios recovered by retry (S2, S3, S4, S7, S9, S10), 1 safe default (S5), and 1 output guardrail override (S6). A hosted language model can return malformed or incomplete output more than once because generation is probabilistic and prompts do not guarantee schema compliance. Parsing, schema checks, policy-id matching, retries and a safe default prevent malformed or inconsistent notes from silently reaching the downstream queue.

## Part D results

The smoke test completed **11/11 checks**: happy 3/3, escalation 4/4, edge 4/4.

| ID | Category | Expected status | Observed status | Result |
|---|---|---:|---:|---|
| Q1 | happy | 200 | 200 | PASS |
| Q2 | happy | 200 | 200 | PASS |
| Q3 | escalation | 200 | 200 | PASS |
| Q4 | escalation | 200 | 200 | PASS |
| Q5 | edge | 400 | 400 | PASS |
| Q6 | edge | 422 | 422 | PASS |
| Q7 | escalation | 200 | 200 | PASS |
| Q8 | escalation | 200 | 200 | PASS |
| Q9 | edge | 503 | 503 | PASS |
| Q10 | happy | 200 | 200 | PASS |
| Q11 | edge | 200 | 200 | PASS |

Metrics after that run (latency is environment-dependent):

```json
{
  "requests_total": 10,
  "success_total": 7,
  "error_total": 3,
  "success_rate": 0.7,
  "error_rate": 0.3,
  "avg_latency_ms": 9.5143,
  "human_review_total": 5,
  "human_review_rate": 0.5
}
```

Representative committed log lines from the run (timestamps and latency vary):

```text
2026-10-26 09:31:09,555 INFO endpoint=/score status_code=200 latency_ms=12.75 txn_id=T800001 fraud_probability=0.0157
2026-10-26 09:31:09,579 WARNING endpoint=/score status_code=200 latency_ms=10.42 txn_id=T800003 fraud_probability=0.1550
2026-10-26 09:31:09,624 ERROR endpoint=/case status_code=503 latency_ms=11.28 txn_id=T900099 fraud_probability=0.8755
```

The service validates typed request bodies through Pydantic (422) and business constraints before model inference (400). The score path applies the model threshold, then a separate large-payment/new-device rule can hold a payment even when it is not flagged. A flagged payment is never auto-approved; policy-note generation failures return 503 without crashing the service, and a safe default requires manual review.

The separate `T900006` guardrail check also passed: keeping the transaction ID and UPI question but using `amount_inr=5000` with `avg_amount_30d_inr=3000` produces a local-model probability of 0.3013, between this run's 0.3 threshold and 0.5. The mock's first answer is `approve`; the service overrides it to `hold_for_review` and sets both `guardrail_override` and `human_review` to true.

The checked local run used pandas 2.2.3 and scikit-learn 1.8.0 because those are installed in the build workspace; `requirements.txt` pins the assignment's required pandas 3.x and scikit-learn 1.9.x versions. The smoke-test contract passed locally, but the random-forest metrics can differ when the required pinned versions are installed, so rerun `train.py` and copy its printed results if your installed environment changes the measured metrics.

## Limitations

- The dataset is synthetic and generated from a hand-built probability rule, so performance is not evidence of production fraud detection quality.
- The TF-IDF retriever uses lexical overlap and may miss semantically equivalent policy questions.
- `mock_llm.py` is a deterministic scripted stand-in and does not represent the variability or capabilities of a hosted language model.
- The threshold is selected from a coarse nine-value grid and a validation split containing only 73 fraud cases.
- The model uses a limited set of transaction features and does not include real device, network, merchant-history or confirmed chargeback investigations.
- The API uses in-memory metrics and does not provide durable monitoring, authentication, authorization, rate limiting or database persistence.
- The case-note output is validated structurally but still requires human review before any real payment action.
