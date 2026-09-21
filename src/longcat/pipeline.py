from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .io import load_config


KEY_COLUMNS = ["task_id", "fips_code", "county", "state", "issue_time", "target_time"]
PREDICTION_COLUMNS = KEY_COLUMNS + ["predicted_x"]
CATEGORICAL_FEATURES = ["fips_code", "issue_hour", "lead_band"]
NUMERIC_FEATURES = [
    "lead_minutes", "lead_sqrt", "lead_log", "target_hour_sin", "target_hour_cos",
    "target_doy_sin", "target_doy_cos", "denominator_log", "cross_county_last_ratio",
    "last", "staleness_hours", "median_1h", "mean_6h", "max_6h", "median_24h",
    "max_24h", "median_168h", "q90_168h", "q99_168h", "trend_6h",
    "lag_1d", "lag_1d_missing", "lag_7d", "lag_7d_missing", "lag_14d", "lag_14d_missing",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


@dataclass
class Panel:
    series: dict[str, pd.Series]


def county_map(config: dict | None = None) -> dict[str, dict]:
    config = config or load_config()
    return {item["fips_code"]: item for item in config["counties"]}


def load_selected_files(paths: Iterable[Path]) -> Panel:
    pieces: dict[str, list[pd.Series]] = {}
    for path in paths:
        frame = pd.read_csv(path, dtype={"fips_code": str})
        frame["fips_code"] = frame["fips_code"].str.zfill(5)
        frame = frame.rename(columns={"sum": "customers_out"})
        frame["timestamp"] = pd.to_datetime(frame["run_start_time"], utc=True)
        frame = (
            frame[["fips_code", "timestamp", "customers_out"]]
            .dropna()
            .groupby(["fips_code", "timestamp"], as_index=False)["customers_out"].mean()
        )
        for fips, group in frame.groupby("fips_code", sort=False):
            values = group.set_index("timestamp")["customers_out"].sort_index().astype(float)
            pieces.setdefault(fips, []).append(values)
    return Panel({fips: pd.concat(values).sort_index() for fips, values in pieces.items()})


def task_schedule(task_id: str, first: str | pd.Timestamp, last: str | pd.Timestamp) -> pd.DatetimeIndex:
    frequency = "24h" if task_id == "A" else "6h"
    return pd.date_range(pd.Timestamp(first), pd.Timestamp(last), freq=frequency, tz="UTC")


def release_schedule(config: dict | None = None) -> list[tuple[str, pd.Timestamp]]:
    config = config or load_config()
    rows: list[tuple[str, pd.Timestamp]] = []
    for task_id, key in (("A", "task_a"), ("B", "task_b")):
        contract = config["contract"][key]
        rows.extend((task_id, issue) for issue in task_schedule(task_id, contract["first_issue"], contract["last_issue"]))
    return rows


def _asof(series: pd.Series, timestamp: pd.Timestamp, maximum_age: str = "2h") -> float:
    position = series.index.searchsorted(timestamp, side="right") - 1
    if position < 0 or timestamp - series.index[position] > pd.Timedelta(maximum_age):
        return np.nan
    return float(series.iloc[position])


def _state(series: pd.Series, issue_time: pd.Timestamp) -> dict[str, float]:
    position = series.index.searchsorted(issue_time, side="right")
    history = series.iloc[:position]
    if history.empty:
        raise ValueError(f"No causal history at {issue_time}")
    last = float(history.iloc[-1])
    last_time = history.index[-1]

    def window(delta: str) -> np.ndarray:
        values = history[history.index > issue_time - pd.Timedelta(delta)].to_numpy(dtype=float)
        return values if values.size else np.asarray([last], dtype=float)

    one = window("1h")
    six = window("6h")
    day = window("24h")
    week = window("168h")
    return {
        "last": last,
        "staleness_hours": float((issue_time - last_time).total_seconds() / 3600),
        "median_1h": float(np.median(one)),
        "mean_6h": float(np.mean(six)),
        "max_6h": float(np.max(six)),
        "median_24h": float(np.median(day)),
        "max_24h": float(np.max(day)),
        "median_168h": float(np.median(week)),
        "q90_168h": float(np.quantile(week, 0.90)),
        "q99_168h": float(np.quantile(week, 0.99)),
        "trend_6h": float(last - np.mean(six)),
        "max_source_timestamp": last_time,
    }


def make_feature_rows(
    panel: Panel,
    schedule: Iterable[tuple[str, pd.Timestamp]],
    include_target: bool,
    config: dict | None = None,
) -> pd.DataFrame:
    config = config or load_config()
    counties = county_map(config)
    output: list[dict] = []
    for task_id, raw_issue in schedule:
        issue = pd.Timestamp(raw_issue)
        issue = issue.tz_localize("UTC") if issue.tzinfo is None else issue.tz_convert("UTC")
        task_contract = config["contract"]["task_a" if task_id == "A" else "task_b"]
        states = {fips: _state(panel.series[fips], issue) for fips in counties}
        cross = float(np.median([states[fips]["last"] / counties[fips]["customers"] for fips in counties]))
        for fips, county in counties.items():
            series = panel.series[fips]
            state = states[fips]
            for lead in task_contract["lead_minutes"]:
                target = issue + pd.Timedelta(minutes=int(lead))
                lag_1_time = target - pd.Timedelta("1d")
                lag_1 = _asof(series, lag_1_time) if lag_1_time <= issue else np.nan
                lag_7 = _asof(series, target - pd.Timedelta("7d"))
                lag_14 = _asof(series, target - pd.Timedelta("14d"))
                record = {
                    "task_id": task_id,
                    "fips_code": fips,
                    "county": county["county"],
                    "state": county["state"],
                    "issue_time": issue,
                    "target_time": target,
                    "issue_hour": str(issue.hour),
                    "lead_band": str((int(lead) - 1) // 360),
                    "lead_minutes": int(lead),
                    "lead_sqrt": float(np.sqrt(lead)),
                    "lead_log": float(np.log1p(lead)),
                    "target_hour_sin": float(np.sin(2 * np.pi * target.hour / 24)),
                    "target_hour_cos": float(np.cos(2 * np.pi * target.hour / 24)),
                    "target_doy_sin": float(np.sin(2 * np.pi * target.dayofyear / 365.25)),
                    "target_doy_cos": float(np.cos(2 * np.pi * target.dayofyear / 365.25)),
                    "denominator_log": float(np.log(county["customers"])),
                    "cross_county_last_ratio": cross,
                    **{key: value for key, value in state.items() if key != "max_source_timestamp"},
                    "lag_1d": float(lag_1 if np.isfinite(lag_1) else state["median_168h"]),
                    "lag_1d_missing": int(not np.isfinite(lag_1)),
                    "lag_7d": float(lag_7 if np.isfinite(lag_7) else state["median_168h"]),
                    "lag_7d_missing": int(not np.isfinite(lag_7)),
                    "lag_14d": float(lag_14 if np.isfinite(lag_14) else state["median_168h"]),
                    "lag_14d_missing": int(not np.isfinite(lag_14)),
                    "max_source_timestamp": state["max_source_timestamp"],
                }
                if include_target:
                    if task_id == "A":
                        values = series[(series.index > target - pd.Timedelta("1h")) & (series.index <= target)].to_numpy()
                        record["observed_customers_out"] = float(np.mean(values)) if len(values) >= 2 else np.nan
                    else:
                        record["observed_customers_out"] = float(series.loc[target]) if target in series.index else np.nan
                output.append(record)
    frame = pd.DataFrame(output)
    assert (pd.to_datetime(frame["max_source_timestamp"], utc=True) <= pd.to_datetime(frame["issue_time"], utc=True)).all()
    return frame


def fit_models(frame: pd.DataFrame, config: dict | None = None) -> dict:
    config = config or load_config()
    artifact = {"feature_columns": MODEL_FEATURES, "models": {}, "blend": {}}
    for task_id in ("A", "B"):
        task = frame[(frame["task_id"] == task_id) & frame["observed_customers_out"].notna()].copy()
        transformer = ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ])
        estimator = Pipeline([
            ("transform", transformer),
            ("ridge", Ridge(alpha=float(config["model"]["ridge_alpha"]))),
        ])
        weights = 1.0 + np.minimum(np.log1p(task["observed_customers_out"].to_numpy()) / 10.0, 1.5)
        estimator.fit(task[MODEL_FEATURES], np.log1p(task["observed_customers_out"].to_numpy()), ridge__sample_weight=weights)
        artifact["models"][task_id] = estimator
        artifact["blend"][task_id] = float(config["model"][f"task_{task_id.lower()}_last_state_blend"])
        artifact.setdefault("training_rows", {})[task_id] = int(len(task))
    return artifact


def predict_counts(model_artifact: dict, frame: pd.DataFrame, apply_calibration: bool = True) -> np.ndarray:
    prediction = np.empty(len(frame), dtype=float)
    for task_id in ("A", "B"):
        mask = frame["task_id"].eq(task_id).to_numpy()
        if not mask.any():
            continue
        structural = np.maximum(0.0, np.expm1(model_artifact["models"][task_id].predict(frame.loc[mask, MODEL_FEATURES])))
        blend = float(model_artifact["blend"][task_id])
        prediction[mask] = (1.0 - blend) * structural + blend * frame.loc[mask, "last"].to_numpy(dtype=float)
    if apply_calibration and model_artifact.get("calibration"):
        for task_id, specification in model_artifact["calibration"].items():
            task_mask = frame["task_id"].eq(task_id).to_numpy()
            if not task_mask.any():
                continue
            task = frame.loc[task_mask]
            keys = task[specification["group_columns"]].astype(str).agg("|".join, axis=1)
            scale = keys.map(specification["scales"]).fillna(specification["global_scale"]).to_numpy(dtype=float)
            prediction[task_mask] *= scale
    return prediction


def _weighted_median_scale(
    truth: np.ndarray,
    prediction: np.ndarray,
    denominator: np.ndarray,
    scale_bounds: tuple[float, float],
) -> float:
    valid = np.isfinite(truth) & np.isfinite(prediction) & (prediction > 0) & (denominator > 0)
    if not valid.any():
        return 1.0
    ratios = truth[valid] / prediction[valid]
    weights = prediction[valid] / denominator[valid]
    order = np.argsort(ratios)
    ratios, weights = ratios[order], weights[order]
    index = int(np.searchsorted(np.cumsum(weights), weights.sum() / 2.0))
    return float(np.clip(ratios[min(index, len(ratios) - 1)], *scale_bounds))


def fit_calibration(model_artifact: dict, frame: pd.DataFrame, config: dict | None = None) -> dict:
    """Fit a pre-declared groupwise MAE calibrator on an out-of-fold frame."""
    config = config or load_config()
    base_prediction = predict_counts(model_artifact, frame, apply_calibration=False)
    denominators = frame["fips_code"].map({x["fips_code"]: x["customers"] for x in config["counties"]}).to_numpy(dtype=float)
    truth = frame["observed_customers_out"].to_numpy(dtype=float)
    calibration = {}
    declared = config["model"]["calibration"]
    scale_bounds = tuple(float(value) for value in declared["scale_bounds"])
    declarations = {
        task_id: {
            "group_columns": list(declared[f"task_{task_id.lower()}_groups"]),
            "shrinkage_rows": float(declared[f"task_{task_id.lower()}_shrinkage_rows"]),
        }
        for task_id in ("A", "B")
    }
    for task_id, declaration in declarations.items():
        mask = frame["task_id"].eq(task_id).to_numpy()
        task = frame.loc[mask].reset_index(drop=True)
        task_truth = truth[mask]
        task_prediction = base_prediction[mask]
        task_denominator = denominators[mask]
        global_scale = _weighted_median_scale(
            task_truth, task_prediction, task_denominator, scale_bounds
        )
        keys = task[declaration["group_columns"]].astype(str).agg("|".join, axis=1)
        scales = {}
        for key, positions in keys.groupby(keys).groups.items():
            positions = np.asarray(list(positions), dtype=int)
            local = _weighted_median_scale(
                task_truth[positions],
                task_prediction[positions],
                task_denominator[positions],
                scale_bounds,
            )
            n = float(len(positions))
            shrinkage = float(declaration["shrinkage_rows"])
            scales[str(key)] = float((n * local + shrinkage * global_scale) / (n + shrinkage))
        calibration[task_id] = {
            **declaration,
            "global_scale": global_scale,
            "scales": scales,
            "fitted_rows": int(mask.sum()),
            "objective": "weighted median scale minimizing outage-ratio MAE",
        }
    artifact = dict(model_artifact)
    artifact["calibration"] = calibration
    return artifact


def submission_from_features(model_artifact: dict, frame: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    config = config or load_config()
    counties = county_map(config)
    counts = predict_counts(model_artifact, frame)
    denominators = frame["fips_code"].map({key: value["customers"] for key, value in counties.items()}).to_numpy(dtype=float)
    result = frame[KEY_COLUMNS].copy()
    result["predicted_x"] = np.clip(counts / denominators, 0.0, 1.0)
    for column in ("issue_time", "target_time"):
        result[column] = pd.to_datetime(result[column], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return result[PREDICTION_COLUMNS]


def metric_bundle(frame: pd.DataFrame, predicted_counts: np.ndarray, config: dict | None = None) -> dict[str, float]:
    config = config or load_config()
    denominators = frame["fips_code"].map({x["fips_code"]: x["customers"] for x in config["counties"]}).to_numpy(dtype=float)
    truth = frame["observed_customers_out"].to_numpy(dtype=float) / denominators
    prediction = predicted_counts / denominators
    absolute = np.abs(truth - prediction)
    tail = frame["observed_customers_out"].to_numpy(dtype=float) >= np.quantile(frame["observed_customers_out"], 0.90)
    return {
        "rows": int(len(frame)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean((truth - prediction) ** 2))),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "top_decile_mae": float(np.mean(absolute[tail])),
        "bias": float(np.mean(prediction - truth)),
        "pearson": float(np.corrcoef(truth, prediction)[0, 1]),
    }
