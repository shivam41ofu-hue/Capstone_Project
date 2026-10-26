"""Audit, clean, and feature-engineer the synthetic transaction export."""
from pathlib import Path
import pandas as pd

RAW_PATH = Path("transactions_raw.csv")
CATEGORICAL = ["merchant_category", "city", "payment_mode"]
EXCLUDED = ["txn_id", "customer_id", "chargeback_filed", "is_fraud"]


def clean_transactions(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy; never mutate the caller's DataFrame."""
    df = raw_df.copy(deep=True)
    df = df.drop_duplicates()
    for col in ["merchant_category", "city"]:
        df[col] = df[col].astype("string").str.strip().str.title()
    df["amount_inr"] = pd.to_numeric(df["amount_inr"], errors="coerce")
    df = df.dropna(subset=["amount_inr"])
    df = df.loc[df["amount_inr"].between(1, 1_000_000)]
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the three required model features."""
    out = df.copy(deep=True)
    out["amount_vs_avg_ratio"] = out["amount_inr"] / out["avg_amount_30d_inr"]
    out["is_night"] = (out["hour_of_day"] <= 5).astype(int)
    out["high_value"] = (out["amount_inr"] > 50_000).astype(int)
    return out


def audit(raw: pd.DataFrame) -> dict:
    dedup = raw.drop_duplicates()
    amount = pd.to_numeric(dedup["amount_inr"], errors="coerce")
    distance = pd.to_numeric(dedup["distance_from_home_km"], errors="coerce")
    report = {
        "raw_rows": len(raw), "raw_columns": len(raw.columns),
        "exact_duplicates": int(raw.duplicated().sum()),
        "missing_amount_after_dedup": int(amount.isna().sum()),
        "missing_distance_after_dedup": int(distance.isna().sum()),
        "out_of_range_amount_after_dedup": int(((amount < 1) | (amount > 1_000_000)).sum()),
        "merchant_spellings": int(raw["merchant_category"].nunique(dropna=False)),
        "city_spellings": int(raw["city"].nunique(dropna=False)),
    }
    return report


def main() -> None:
    raw = pd.read_csv(RAW_PATH)
    a = audit(raw)
    print("AUDIT")
    print(pd.DataFrame([a]).to_string(index=False))
    clean = clean_transactions(raw)
    featured = add_features(clean)
    print("\nCLEANING CHECKPOINTS")
    print(f"clean_rows={len(clean)}")
    print("merchant_category_counts:")
    print(clean["merchant_category"].value_counts().to_string())
    print("city_counts:")
    print(clean["city"].value_counts().to_string())
    print("payment_mode_counts:")
    print(clean["payment_mode"].value_counts().to_string())
    print(f"amount_min={clean['amount_inr'].min():.2f}")
    print(f"amount_max={clean['amount_inr'].max():.2f}")
    print(f"missing_distance_remaining={clean['distance_from_home_km'].isna().sum()}")
    print(f"is_night_sum={featured['is_night'].sum()}")
    print(f"high_value_sum={featured['high_value'].sum()}")
    print(f"amount_vs_avg_ratio_max={featured['amount_vs_avg_ratio'].max():.2f}")
    print("\nLEAKAGE TABLE")
    print(clean.groupby("chargeback_filed")["is_fraud"].agg(rows="size", fraud_rows="sum").to_string())
    fraud = int(clean["is_fraud"].sum())
    print(f"fraud_rows={fraud}; fraud_rate={fraud/len(clean):.4%}")
    print(f"never_fraud_accuracy={(clean['is_fraud'].eq(0).mean()):.4%}; never_fraud_recall=0")
    print("excluded columns:", EXCLUDED)
    print("model inputs before one-hot:", [c for c in featured.columns if c not in EXCLUDED])
    # One-row and non-mutation checks.
    one = raw.iloc[[0]].copy()
    original = one.copy(deep=True)
    clean_transactions(one)
    add_features(one)
    assert one.equals(original), "clean_transactions/add_features mutated their input"
    print("one_row_and_non_mutation_checks=PASS")

if __name__ == "__main__":
    main()
