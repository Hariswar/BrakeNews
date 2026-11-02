"""Recall-risk model: how likely is a covering recall within 24 months, for a vehicle x theme
pair as of a month? Trained on features.py's table; outputs in data/clean/model/ and models/.

Splits are by time, never at random: a random split would put a pair's March row in training
and its April row in the test set. Rows are dated by month and the label looks HORIZON_MONTHS
ahead, so each block is separated from the next by that horizon; otherwise a training label
would already know about a recall inside the later block.
    train  ...........  gap  validation (3 years)  gap  test (4 years) = last labelled months
Validation picks the number of boosting rounds (early stopping); the test block is scored once.

The baseline is the signals.py rule (3+ complaints, PRR >= 2, chi-square >= 4). The headline
comparison holds the alert volume fixed: if analysts can review as many alerts as the rule
raises, how many of those alerts does each method spend on pairs that are then recalled?

Scores are LightGBM probabilities without class reweighting, so they stay roughly calibrated
(metrics.json reports the mean score next to the observed rate on the test block). After
evaluation the model is retrained on every labelled row with the chosen number of rounds and
scores each pair's latest month in the last WINDOW_MONTHS months.

Every run is logged to MLflow in mlruns/ (a SQLite store, since MLflow 3 retired the plain
folder store); browse with `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.

Outputs:
  models/recall_risk.txt                the final LightGBM model
  data/clean/model/metrics.json         validation and test metrics, baselines, importance
  data/clean/model/risk_scores.parquet  current pairs with their risk score
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import CLEAN, ROOT
from pipeline.features import CATEGORICAL, FEATURES, HORIZON_MONTHS
from pipeline.signals import WINDOW_MONTHS

TEST_MONTHS = 48
VALIDATION_MONTHS = 36
PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 200,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "seed": 0, "verbose": -1, "num_threads": 4,
}
MAX_ROUNDS = 2000
MODEL_FEATURES = FEATURES


def splits(mi: pd.Series, last_labelled: int, test_months: int = TEST_MONTHS,
           validation_months: int = VALIDATION_MONTHS,
           gap: int = HORIZON_MONTHS) -> dict[str, pd.Series]:
    test_start = last_labelled - test_months + 1
    val_end = test_start - gap - 1
    val_start = val_end - validation_months + 1
    return {"train": mi < val_start - gap, "validation": mi.between(val_start, val_end),
            "test": mi >= test_start}


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


def _score_current(booster, df: pd.DataFrame, categories: dict, features: tuple[str, ...],
                   models_dir: Path, out: Path) -> int:
    """Save the model trained on every labelled row and score each pair's latest month in
    the last WINDOW_MONTHS months."""
    models_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(models_dir / "recall_risk.txt"))
    last_mi = int(df["mi"].max())
    latest = (df[df["mi"] > last_mi - WINDOW_MONTHS].sort_values("mi")
              .groupby(["make", "model", "model_year", "theme_id"], observed=True).tail(1))
    scores = (latest[["make", "model", "model_year", "theme_id", "month", "a", "prr", "signal"]]
              .assign(risk=booster.predict(_frame(latest, categories, features)))
              .sort_values("risk", ascending=False))
    scores.to_parquet(out / "risk_scores.parquet", index=False)
    return len(scores)


def run(clean: Path = CLEAN, models_dir: Path = ROOT / "models",
        runs_dir: Path = ROOT / "mlruns", max_rounds: int = MAX_ROUNDS,
        params: dict | None = None, test_months: int = TEST_MONTHS,
        validation_months: int = VALIDATION_MONTHS,
        features: tuple[str, ...] = MODEL_FEATURES, final: bool = True) -> dict:
    """Train, evaluate on validation and test, log to MLflow. With final=False (for comparing
    variants) the retrain on all rows and the scoring of current pairs are skipped."""
    import lightgbm as lgb  # pip install -e ".[model]"
    import mlflow

    params = {**PARAMS, **(params or {})}
    out = clean / "model"
    df = pd.read_parquet(out / "training.parquet")
    categories = {c: sorted(df[c].dropna().unique().tolist())
                  for c in CATEGORICAL if c in features}
    labelled = df[df["label"].notna()].reset_index(drop=True)
    parts = splits(labelled["mi"], int(labelled["mi"].max()), test_months, validation_months)
    sets = {name: labelled[mask] for name, mask in parts.items()}
    for name, part in sets.items():
        if part.empty or part["label"].nunique() < 2:
            raise SystemExit(f"[model] the {name} block has no positives or no negatives")

    def dataset(part, reference=None):
        return lgb.Dataset(_frame(part, categories, features), part["label"].astype(int),
                           categorical_feature=list(categories), reference=reference,
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
                                 booster.predict(_frame(part, categories, features),
                                                 num_iteration=rounds),
                                 part["signal"].to_numpy(int), part["prr"].to_numpy())
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
            metrics["scored_pairs"] = _score_current(lgb.train(
                params, dataset(labelled), num_boost_round=rounds), df, categories, features,
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
