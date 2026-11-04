"""Recall-risk model: how likely is a covering recall within 24 months, for a vehicle x theme
pair as of a month? Trained on features.py's table; outputs in data/clean/model/ and models/.

Splits are by time, never at random: a random split would put a pair's March row in training
and its April row in the test set. Rows are dated by month and the label looks HORIZON_MONTHS
ahead, so each block is separated from the next by that horizon; otherwise a training label
would already know about a recall inside the later block.
    train  ...........  gap  validation (3 years)  gap  test (4 years) = last labelled months
Validation picks the number of boosting rounds (early stopping); the test block is scored once.

Choosing features and parameters (`make select`). One validation block is a noisy judge: three
years of one era. So candidates are compared on SELECTION_FOLDS earlier folds instead, each the
same train / gap / validation / gap / test layout shifted back in steps of TEST_MONTHS, the
latest ending where the main validation block ends. No fold's labels reach into the final test
block. The candidate with the best mean average precision over the folds' test blocks wins
(ties: fewer features). The result is fixed below as MODEL_FEATURES and PARAMS, so `make model`
does not depend on a previous run; `make select` says if a rerun disagrees.

Selection result (2026-11-02, selection.json, on quarterly snapshots with Claude-checked recall
matches): dropping the identity features `make` and `theme_id` wins again (mean AP over the
folds 0.0246 against 0.0224 with them). They let the model memorise how often each make and
theme was recalled in the training era, which drifts; theme_recalls and make_recalls_5y carry
the same information, counted as of each snapshot. With and without the surge features is
within noise (0.0246 and 0.0242); the rule picks the higher, so they stay in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import CLEAN, ROOT
from pipeline.features import CATEGORICAL, FEATURES, HORIZON_MONTHS

TEST_MONTHS = 48
VALIDATION_MONTHS = 36
SELECTION_FOLDS = 3
_COMMON = {"objective": "binary", "feature_fraction": 0.8, "bagging_fraction": 0.8,
           "bagging_freq": 1, "lambda_l2": 1.0, "seed": 0, "verbose": -1, "num_threads": 4}
PARAM_SETS = {
    "fast": {**_COMMON, "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200},
    "steady": {**_COMMON, "learning_rate": 0.02, "num_leaves": 15, "min_data_in_leaf": 500},
}
MAX_ROUNDS = 2000

IDENTITY = ("make", "theme_id")
SURGE = ("vehicle_k3", "expected3", "surge_ratio", "surge_logp", "surge", "surge_months")
FEATURE_SETS = {
    "all": FEATURES,
    "no_surge": tuple(f for f in FEATURES if f not in SURGE),
    "no_identity": tuple(f for f in FEATURES if f not in IDENTITY),
    "no_identity_no_surge": tuple(f for f in FEATURES if f not in IDENTITY + SURGE),
}
# Chosen by `make select` (see the module docstring).
MODEL_FEATURE_SET = "no_identity"
MODEL_PARAM_SET = "steady"
MODEL_FEATURES = FEATURE_SETS[MODEL_FEATURE_SET]
PARAMS = PARAM_SETS[MODEL_PARAM_SET]


def splits(mi: pd.Series, last_labelled: int, test_months: int = TEST_MONTHS,
           validation_months: int = VALIDATION_MONTHS,
           gap: int = HORIZON_MONTHS) -> dict[str, pd.Series]:
    test_start = last_labelled - test_months + 1
    val_end = test_start - gap - 1
    val_start = val_end - validation_months + 1
    return {"train": mi < val_start - gap, "validation": mi.between(val_start, val_end),
            "test": mi.between(test_start, last_labelled)}


def selection_ends(last_labelled: int, folds: int = SELECTION_FOLDS,
                   test_months: int = TEST_MONTHS, gap: int = HORIZON_MONTHS) -> list[int]:
    """Last month of each selection fold's test block, newest first: the newest ends where the
    main validation block ends, so no fold label looks past the final test block's start."""
    newest = last_labelled - test_months - gap
    return [newest - i * test_months for i in range(folds)]


def _frame(df: pd.DataFrame, categories: dict[str, list],
           features: tuple[str, ...]) -> pd.DataFrame:
    x = df[list(features)].copy()
    for c in categories:
        x[c] = pd.Categorical(x[c], categories=categories[c])
    return x


def evaluate(y: np.ndarray, score: np.ndarray, rule: np.ndarray, prr: np.ndarray) -> dict:
    """Ranking quality, and precision/recall at the rule's alert volume."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    n_flag = int(rule.sum())
    top = np.argsort(-score, kind="stable")[:n_flag]
    hits_model, hits_rule, positives = int(y[top].sum()), int(y[rule == 1].sum()), int(y.sum())
    ranked = 0 < positives < len(y)
    return {
        "rows": len(y), "positives": positives, "base_rate": positives / len(y),
        "average_precision": average_precision_score(y, score) if ranked else None,
        "roc_auc": roc_auc_score(y, score) if ranked else None,
        "prr_average_precision": average_precision_score(y, prr) if ranked else None,
        "rule_alerts": n_flag,
        "rule_precision": hits_rule / n_flag if n_flag else None,
        "rule_recall": hits_rule / positives if positives else None,
        "model_precision_at_rule_volume": hits_model / n_flag if n_flag else None,
        "model_recall_at_rule_volume": hits_model / positives if positives else None,
        "mean_score": float(score.mean()),
    }


def _track(runs: Path, experiment: str = "recall-risk") -> None:
    import mlflow

    runs.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + (runs / "mlflow.db").as_posix())
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(runs / "artifacts").as_uri())
    mlflow.set_experiment(experiment)


def _score_current(booster, current: pd.DataFrame, categories: dict,
                   features: tuple[str, ...], models_dir: Path, out: Path) -> int:
    """Save the model trained on every labelled row and score the last snapshot: every pair
    with a complaint in the window ending at the last month of data, as of that month. Pairs a
    covering recall has settled have no snapshot after it (features.py), so none is scored."""
    models_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(models_dir / "recall_risk.txt"))
    if "recall_mi" in current:
        current = current[current["recall_mi"].isna()]
    scores = (current[["make", "model", "model_year", "theme_id", "month", "a", "k3",
                       "quiet_months", "prr", "signal", "expected3", "surge_logp", "surge"]]
              .assign(risk=booster.predict(_frame(current, categories, features)))
              .sort_values("risk", ascending=False))
    scores.to_parquet(out / "risk_scores.parquet", index=False)
    return len(scores)


def _fit(labelled: pd.DataFrame, last_labelled: int, categories: dict,
         features: tuple[str, ...], params: dict, max_rounds: int, test_months: int,
         validation_months: int) -> tuple:
    """Train on one train / validation / test layout ending at last_labelled; early-stop on
    validation and evaluate validation and test. Returns (booster, metrics)."""
    import lightgbm as lgb  # pip install -e ".[model]"

    parts = splits(labelled["mi"], last_labelled, test_months, validation_months)
    sets = {name: labelled[mask] for name, mask in parts.items()}
    for name, part in sets.items():
        if part.empty or part["label"].nunique() < 2:
            raise SystemExit(f"[model] the {name} block ending {last_labelled} has no positives "
                             f"or no negatives")
    cats = {c: v for c, v in categories.items() if c in features}

    def dataset(part, reference=None):
        return lgb.Dataset(_frame(part, cats, features), part["label"].astype(int),
                           categorical_feature=list(cats), reference=reference,
                           free_raw_data=False)

    train = dataset(sets["train"])
    booster = lgb.train(params, train, num_boost_round=max_rounds,
                        valid_sets=[dataset(sets["validation"], train)],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    rounds = booster.best_iteration or max_rounds
    metrics: dict = {"rounds": rounds}
    for name in ("train", "validation", "test"):
        metrics[f"{name}_months"] = (f"{str(sets[name]['month'].min())[:7]} to "
                                     f"{str(sets[name]['month'].max())[:7]}")
    for name in ("validation", "test"):
        part = sets[name]
        metrics[name] = evaluate(part["label"].to_numpy(int),
                                 booster.predict(_frame(part, cats, features),
                                                 num_iteration=rounds),
                                 part["signal"].to_numpy(int), part["prr"].to_numpy())
    return booster, metrics


def _load(clean: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(the last snapshot, every labelled row, category levels). Reads only the columns the
    model uses, with floats as float32: the snapshot table has millions of rows."""
    import pyarrow.parquet as pq

    path = clean / "model" / "training.parquet"
    have = set(pq.read_schema(path).names)
    keep = ["make", "model", "model_year", "theme_id", "month", "mi", "label", "recall_mi",
            *FEATURES]
    table = pq.read_table(path, columns=[c for c in dict.fromkeys(keep) if c in have])
    df = table.to_pandas()
    del table
    floats = df.select_dtypes("float64").columns.drop("label", errors="ignore")
    df[floats] = df[floats].astype(np.float32)
    categories = {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL}
    current = df[df["mi"] == df["mi"].max()].reset_index(drop=True)
    labelled = df[df["label"].notna()].reset_index(drop=True)
    return current, labelled, categories


def run(clean: Path = CLEAN, models_dir: Path = ROOT / "models",
        runs_dir: Path = ROOT / "mlruns", max_rounds: int = MAX_ROUNDS,
        params: dict | None = None, test_months: int = TEST_MONTHS,
        validation_months: int = VALIDATION_MONTHS,
        features: tuple[str, ...] = MODEL_FEATURES, final: bool = True) -> dict:
    """Train, evaluate on validation and test, log to MLflow. With final=False (for comparing
    variants) the retrain on all rows and the scoring of current pairs are skipped."""
    import lightgbm as lgb
    import mlflow

    params = {**PARAMS, **(params or {})}
    out = clean / "model"
    current, labelled, categories = _load(clean)
    booster, metrics = _fit(labelled, int(labelled["mi"].max()), categories, features, params,
                            max_rounds, test_months, validation_months)
    rounds = metrics["rounds"]
    metrics["features"] = list(features)
    gain = booster.feature_importance("gain", iteration=rounds)
    metrics["feature_importance"] = {
        f: round(float(g / max(gain.sum(), 1e-9)), 4)
        for f, g in sorted(zip(features, gain, strict=True), key=lambda t: -t[1])}

    _track(runs_dir)
    with mlflow.start_run():
        mlflow.log_params({**params, "rounds": rounds, "horizon_months": HORIZON_MONTHS,
                           "test_months": test_months, "validation_months": validation_months,
                           "features": ",".join(features), "final": final})
        for name in ("validation", "test"):
            mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics[name].items()
                                if isinstance(v, int | float)})
        if final:
            cats = {c: v for c, v in categories.items() if c in features}
            everything = lgb.Dataset(_frame(labelled, cats, features),
                                     labelled["label"].astype(int),
                                     categorical_feature=list(cats))
            metrics["scored_pairs"] = _score_current(
                lgb.train(params, everything, num_boost_round=rounds), current, cats, features,
                models_dir, out)
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
            mlflow.log_artifact(str(out / "metrics.json"))
            mlflow.log_artifact(str(models_dir / "recall_risk.txt"))

    name = "test" if final else "validation"  # variants are compared on validation only
    t = metrics[name]
    print(f"[model] {rounds} rounds; {name} {metrics[f'{name}_months']}: {t['rows']:,} rows, "
          f"{t['positives']:,} followed by a recall ({t['base_rate']:.2%})")
    print(f"[model]   average precision {t['average_precision']:.3f} "
          f"(PRR alone {t['prr_average_precision']:.3f}), ROC AUC {t['roc_auc']:.3f}")
    print(f"[model]   at the rule's {t['rule_alerts']:,} alerts: model precision "
          f"{t['model_precision_at_rule_volume']:.2%}, recall "
          f"{t['model_recall_at_rule_volume']:.1%}; rule {t['rule_precision']:.2%}, "
          f"{t['rule_recall']:.1%}")
    if final:
        print(f"[model] scored {metrics['scored_pairs']:,} current pairs -> "
              f"{out / 'risk_scores.parquet'}")
    return metrics


def select(clean: Path = CLEAN, runs_dir: Path = ROOT / "mlruns",
           feature_sets: dict[str, tuple[str, ...]] | None = None,
           param_sets: dict[str, dict] | None = None, folds: int = SELECTION_FOLDS,
           max_rounds: int = MAX_ROUNDS, test_months: int = TEST_MONTHS,
           validation_months: int = VALIDATION_MONTHS) -> dict:
    """Compare every feature set x parameter set on the selection folds (never the final test
    block); write selection.json and log each fold to MLflow. Returns the summary."""
    import mlflow

    feature_sets = feature_sets or FEATURE_SETS
    param_sets = param_sets or PARAM_SETS
    _, labelled, categories = _load(clean)
    ends = selection_ends(int(labelled["mi"].max()), folds, test_months)
    _track(runs_dir, "recall-risk-selection")
    results = []
    for fname, features in feature_sets.items():
        for pname, params in param_sets.items():
            fold_metrics = []
            for end in ends:
                _, m = _fit(labelled, end, categories, features, params, max_rounds,
                            test_months, validation_months)
                fold_metrics.append({"test_months": m["test_months"], "rounds": m["rounds"],
                                     **{k: m["test"][k] for k in (
                                         "average_precision", "roc_auc", "base_rate",
                                         "prr_average_precision",
                                         "model_precision_at_rule_volume", "rule_precision")}})
                with mlflow.start_run(run_name=f"{fname}/{pname}/{m['test_months']}"):
                    mlflow.log_params({**params, "feature_set": fname, "param_set": pname,
                                       "fold_test_months": m["test_months"],
                                       "rounds": m["rounds"]})
                    mlflow.log_metrics({f"test_{k}": v for k, v in m["test"].items()
                                        if isinstance(v, int | float)})
            results.append({
                "feature_set": fname, "param_set": pname, "n_features": len(features),
                "mean_average_precision": float(np.mean(
                    [f["average_precision"] for f in fold_metrics])),
                "mean_roc_auc": float(np.mean([f["roc_auc"] for f in fold_metrics])),
                "folds": fold_metrics})
            r = results[-1]
            print(f"[select] {fname:22s} {pname:7s} mean AP {r['mean_average_precision']:.4f} "
                  f"AUC {r['mean_roc_auc']:.3f}  (" + ", ".join(
                      f"{f['average_precision']:.4f}" for f in fold_metrics) + ")", flush=True)
    results.sort(key=lambda r: (-round(r["mean_average_precision"], 4), r["n_features"]))
    best = results[0]
    summary = {"winner": {"feature_set": best["feature_set"], "param_set": best["param_set"]},
               "chosen_in_code": {"feature_set": MODEL_FEATURE_SET, "param_set": MODEL_PARAM_SET},
               "fold_test_blocks": [f["test_months"] for f in best["folds"]],
               "candidates": results}
    (clean / "model" / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[select] winner: {best['feature_set']} / {best['param_set']}")
    if (best["feature_set"], best["param_set"]) != (MODEL_FEATURE_SET, MODEL_PARAM_SET):
        print(f"[select] note: model.py trains {MODEL_FEATURE_SET} / {MODEL_PARAM_SET}; update "
              f"MODEL_FEATURE_SET and MODEL_PARAM_SET to use the winner")
    return summary
