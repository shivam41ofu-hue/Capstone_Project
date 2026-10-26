"""Train, compare, tune, and evaluate fraud models reproducibly."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from prepare import clean_transactions, add_features, EXCLUDED

RAW_PATH = Path("transactions_raw.csv")
SEED = 42


def load_feature_table():
    raw = pd.read_csv(RAW_PATH)
    df = add_features(clean_transactions(raw))
    y = df["is_fraud"].astype(int)
    x = df.drop(columns=EXCLUDED)
    # The task specifies get_dummies' natural alphabetical ordering.
    x = pd.get_dummies(x, columns=["merchant_category", "city", "payment_mode"], dtype=int)
    return x, y, df


def split_and_fill(x, y):
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.4, stratify=y, random_state=SEED)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SEED)
    median = float(x_train["distance_from_home_km"].median())
    sets = []
    for part in (x_train, x_val, x_test):
        part = part.copy()
        part["distance_from_home_km"] = part["distance_from_home_km"].fillna(median)
        sets.append(part)
    return (*sets, y_train, y_val, y_test, median)


def make_logistic(class_weight=None):
    return Pipeline([("scale", StandardScaler()),
                     ("model", LogisticRegression(max_iter=1000, class_weight=class_weight))])


def metric_row(name, model, x, y, threshold=0.5):
    prob = model.predict_proba(x)[:, 1]
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"model": name, "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


def _fit_everything(verbose=True):
    x, y, df = load_feature_table()
    x_train, x_val, x_test, y_train, y_val, y_test, median = split_and_fill(x, y)
    if verbose:
        print("SPLITS")
        for label, xx, yy in [("train", x_train, y_train), ("validation", x_val, y_val), ("test", x_test, y_test)]:
            print(f"{label}: rows={len(xx)}, fraud={int(yy.sum())}")
        print(f"training_distance_median={median}")
        print("INPUT_COLUMNS", json.dumps(list(x.columns)))
    # Three imbalance approaches.
    plain = make_logistic()
    balanced = make_logistic(class_weight="balanced")
    plain.fit(x_train, y_train)
    balanced.fit(x_train, y_train)
    train_aug = x_train.copy()
    train_aug["__target__"] = y_train.to_numpy()
    negatives = train_aug[train_aug["__target__"] == 0]
    positives = train_aug[train_aug["__target__"] == 1]
    sampled_fraud = positives.sample(n=len(negatives), replace=True, random_state=42)
    oversampled = pd.concat([negatives, sampled_fraud], ignore_index=True)
    over_model = make_logistic()
    over_model.fit(oversampled.drop(columns="__target__"), oversampled["__target__"])
    if verbose:
        rows = [metric_row("plain", plain, x_val, y_val),
                metric_row("balanced", balanced, x_val, y_val),
                metric_row("random_oversampling", over_model, x_val, y_val)]
        print("VALIDATION_IMBALANCE_RESULTS")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda z: f"{z:.3f}"))
        print(f"oversampled_training_rows={len(oversampled)}; nonfraud={int((oversampled['__target__']==0).sum())}; fraud={int((oversampled['__target__']==1).sum())}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    balanced_cv = cross_val_score(make_logistic(class_weight="balanced"), x_train, y_train, cv=cv, scoring="f1")
    base_forest = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    forest_cv = cross_val_score(base_forest, x_train, y_train, cv=cv, scoring="f1")
    grid = GridSearchCV(RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
                        {"max_depth": [4, 6, 8, None], "min_samples_leaf": [5, 10, 20]},
                        scoring="f1", cv=cv, n_jobs=-1, refit=True)
    grid.fit(x_train, y_train)
    model = grid.best_estimator_
    if verbose:
        print("CROSS_VALIDATION")
        print(f"balanced_logistic_f1_mean={balanced_cv.mean():.3f}; std={balanced_cv.std():.3f}")
        print(f"untuned_forest_f1_mean={forest_cv.mean():.3f}; std={forest_cv.std():.3f}")
        best_std = float(grid.cv_results_["std_test_score"][grid.best_index_])
        print(f"grid_combinations={len(grid.cv_results_['params'])}; best_params={grid.best_params_}; best_mean_cv_f1={grid.best_score_:.3f}; best_cv_std={best_std:.3f}")
    # Threshold selection on validation only.
    val_prob = model.predict_proba(x_val)[:, 1]
    threshold_rows = []
    for t in np.arange(0.1, 1.0, 0.1):
        pred = (val_prob >= round(float(t), 1)).astype(int)
        threshold_rows.append({"threshold": round(float(t), 1),
                               "precision": precision_score(y_val, pred, zero_division=0),
                               "recall": recall_score(y_val, pred, zero_division=0),
                               "f1": f1_score(y_val, pred, zero_division=0),
                               "flagged": int(pred.sum())})
    eligible = [r["threshold"] for r in threshold_rows if r["recall"] >= 0.65]
    t_star = max(eligible) if eligible else 0.1
    if verbose:
        print("VALIDATION_THRESHOLDS")
        print(pd.DataFrame(threshold_rows).to_string(index=False, float_format=lambda z: f"{z:.3f}"))
        print(f"t_star={t_star:.1f}")
    test_prob = model.predict_proba(x_test)[:, 1]
    test_pred = (test_prob >= t_star).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()
    if verbose:
        print("FINAL_TEST")
        print(f"precision={precision_score(y_test,test_pred,zero_division=0):.3f}; recall={recall_score(y_test,test_pred,zero_division=0):.3f}; f1={f1_score(y_test,test_pred,zero_division=0):.3f}")
        print(f"confusion_matrix TN={tn}, FP={fp}, FN={fn}, TP={tp}; flagged={int(test_pred.sum())}")
    # Leakage experiment: explicitly append chargeback as 27th column.
    raw = pd.read_csv(RAW_PATH)
    full = add_features(clean_transactions(raw))
    leak_x = full.drop(columns=["txn_id", "customer_id", "is_fraud"])
    leak_x = pd.get_dummies(leak_x, columns=["merchant_category", "city", "payment_mode"], dtype=int)
    # enforce pinned clean columns followed by chargeback_filed
    leak_x = x.reindex(columns=list(x.columns)).copy()
    leak_x["chargeback_filed"] = full["chargeback_filed"].to_numpy()
    leak_train, leak_temp, ly_train, ly_temp = train_test_split(leak_x, y, test_size=.4, stratify=y, random_state=42)
    leak_val, _, ly_val, _ = train_test_split(leak_temp, ly_temp, test_size=.5, stratify=ly_temp, random_state=42)
    leak_median = float(leak_train["distance_from_home_km"].median())
    for part in (leak_train, leak_val):
        part.loc[:, "distance_from_home_km"] = part["distance_from_home_km"].fillna(leak_median)
    leak_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42,
                                        max_depth=grid.best_params_["max_depth"],
                                        min_samples_leaf=grid.best_params_["min_samples_leaf"])
    leak_model.fit(leak_train, ly_train)
    clean_at_half = f1_score(y_val, (model.predict_proba(x_val)[:, 1] >= .5).astype(int))
    leak_f1 = f1_score(ly_val, (leak_model.predict_proba(leak_val)[:, 1] >= .5).astype(int))
    if verbose:
        print(f"LEAKAGE_EXPERIMENT clean_tuned_f1={clean_at_half:.3f}; leaky_f1={leak_f1:.3f}")
        importance = sorted(zip(x.columns, model.feature_importances_), key=lambda z: z[1], reverse=True)
        print("TOP_FEATURE_IMPORTANCE")
        print(pd.DataFrame(importance[:5], columns=["feature", "importance"]).to_string(index=False, float_format=lambda z: f"{z:.4f}"))
    return {"model": model, "columns": list(x.columns), "median": median,
            "threshold": float(t_star), "grid_params": grid.best_params_}


def build_final_model():
    """Rebuild the final model and return the runtime contract used by app.py."""
    return _fit_everything(verbose=False)

if __name__ == "__main__":
    _fit_everything(verbose=True)
