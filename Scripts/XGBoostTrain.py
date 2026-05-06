import pandas as pd
import numpy as np
import yaml
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import json
from pathlib import Path

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from xgboost import XGBClassifier

def clean_feature_names(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.replace(r"[^\w]", "_", regex=True)   # replace special chars
        .str.replace(r"_+", "_", regex=True)      # collapse multiple _
        .str.strip("_")                           # remove leading/trailing _
    )
    return df

def load_config(config_path="XGtrain.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
#-----Helper Function Block
def _to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_list()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    return obj


def _quantile_higher(values, q):
    q = min(max(float(q), 0.0), 1.0)
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))


def compute_conformal_quantile(probs_cal, y_cal, alpha=0.10):
    """
    Split conformal for binary classification using:
        s(x, y) = 1 - p(y|x)
    """
    y_cal_arr = np.asarray(y_cal).astype(int)
    true_class_probs = probs_cal[np.arange(len(y_cal_arr)), y_cal_arr]
    nonconformity_scores = 1.0 - true_class_probs

    m = len(nonconformity_scores)
    q_level = np.ceil((m + 1) * (1 - alpha)) / m
    q_level = min(q_level, 1.0)

    q_alpha = _quantile_higher(nonconformity_scores, q_level)

    return {
        "alpha": float(alpha),
        "q_level": float(q_level),
        "q_alpha": float(q_alpha),
        "nonconformity_scores": nonconformity_scores
    }


def build_conformal_sets(probs, q_alpha):
    """
    For each sample, include class y if 1 - p(y|x) <= q_alpha
    """
    cp_sets = []
    cp_sizes = []
    cp_ambiguous = []

    for row in probs:
        label_set = []
        for y in [0, 1]:
            score = 1.0 - row[y]
            if score <= q_alpha:
                label_set.append(y)

        cp_sets.append(label_set)
        cp_sizes.append(len(label_set))
        cp_ambiguous.append(int(len(label_set) != 1))

    return cp_sets, np.array(cp_sizes), np.array(cp_ambiguous)


def cp_set_to_str(label_set):
    return "{" + ",".join(map(str, label_set)) + "}"


def summarize_conformal(cp_sets, y_true):
    y_true_arr = np.asarray(y_true).astype(int)
    covered = np.array([int(y in s) for y, s in zip(y_true_arr, cp_sets)])
    sizes = np.array([len(s) for s in cp_sets])

    return {
        "coverage": float(covered.mean()),
        "avg_set_size": float(sizes.mean()),
        "singleton_rate": float((sizes == 1).mean()),
        "ambiguous_rate": float((sizes > 1).mean()),
        "empty_set_rate": float((sizes == 0).mean()),
        "covered_vector": covered,
        "set_sizes": sizes
    }


def evaluate_deferral_policy(
    probs,
    y_true,
    base_threshold,
    tau,
    c_fn,
    c_fp,
    c_def,
    cp_sets=None,
    defer_on_ambiguous=True
):
    """
    base_threshold: your current class threshold from Step 4
    tau: certainty threshold for deferral
    """
    y_true_arr = np.asarray(y_true).astype(int)

    p1 = probs[:, 1]
    certainty = probs.max(axis=1)
    pred_base = (p1 >= base_threshold).astype(int)

    defer_mask = certainty < tau

    if cp_sets is not None and defer_on_ambiguous:
        cp_sizes = np.array([len(s) for s in cp_sets])
        defer_mask = defer_mask | (cp_sizes != 1)

    retained_mask = ~defer_mask

    fn = int(((pred_base == 0) & (y_true_arr == 1) & retained_mask).sum())
    fp = int(((pred_base == 1) & (y_true_arr == 0) & retained_mask).sum())
    tp = int(((pred_base == 1) & (y_true_arr == 1) & retained_mask).sum())
    tn = int(((pred_base == 0) & (y_true_arr == 0) & retained_mask).sum())
    deferred = int(defer_mask.sum())

    total_cost = (c_fn * fn) + (c_fp * fp) + (c_def * deferred)
    cost_per_sample = total_cost / len(y_true_arr)

    if retained_mask.sum() > 0:
        retained_error = float((pred_base[retained_mask] != y_true_arr[retained_mask]).mean())
        retained_recall = float(
            ((pred_base == 1) & (y_true_arr == 1) & retained_mask).sum() /
            max(((y_true_arr == 1) & retained_mask).sum(), 1)
        )
        retained_precision = float(
            ((pred_base == 1) & (y_true_arr == 1) & retained_mask).sum() /
            max(((pred_base == 1) & retained_mask).sum(), 1)
        )
    else:
        retained_error = np.nan
        retained_recall = np.nan
        retained_precision = np.nan

    pred_with_deferral = pred_base.copy().astype(int)
    pred_with_deferral[defer_mask] = -1   # -1 means deferred

    return {
        "tau": float(tau),
        "base_threshold": float(base_threshold),
        "coverage": float(retained_mask.mean()),
        "deferred_fraction": float(defer_mask.mean()),
        "deferred_count": deferred,
        "retained_error": retained_error,
        "retained_recall": retained_recall,
        "retained_precision": retained_precision,
        "fn": fn,
        "fp": fp,
        "tp": tp,
        "tn": tn,
        "total_cost": float(total_cost),
        "cost_per_sample": float(cost_per_sample),
        "pred_base": pred_base,
        "pred_with_deferral": pred_with_deferral,
        "certainty": certainty,
        "defer_mask": defer_mask.astype(int),
        "retained_mask": retained_mask.astype(int)
    }


def tune_deferral_threshold(
    probs_cal,
    y_cal,
    base_threshold,
    tau_values,
    c_fn,
    c_fp,
    c_def,
    cp_sets_cal=None,
    defer_on_ambiguous=True
):
    rows = []

    for tau in tau_values:
        res = evaluate_deferral_policy(
            probs=probs_cal,
            y_true=y_cal,
            base_threshold=base_threshold,
            tau=tau,
            c_fn=c_fn,
            c_fp=c_fp,
            c_def=c_def,
            cp_sets=cp_sets_cal,
            defer_on_ambiguous=defer_on_ambiguous
        )

        rows.append({
            "tau": res["tau"],
            "coverage": res["coverage"],
            "deferred_fraction": res["deferred_fraction"],
            "retained_error": res["retained_error"],
            "retained_recall": res["retained_recall"],
            "retained_precision": res["retained_precision"],
            "fn": res["fn"],
            "fp": res["fp"],
            "deferred_count": res["deferred_count"],
            "total_cost": res["total_cost"],
            "cost_per_sample": res["cost_per_sample"]
        })

    df = pd.DataFrame(rows).sort_values(["cost_per_sample", "tau"], ascending=[True, True]).reset_index(drop=True)
    best = df.iloc[0].to_dict()

    return df, best


def save_pipeline_artifacts(cfg, model_name, results_dict, dfs_dict):
    out_dir = Path(cfg["output"]["results_dir"]) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_paths = {}

    for key, df in dfs_dict.items():
        csv_path = out_dir / f"{key}.csv"
        df.to_csv(csv_path, index=False)
        artifact_paths[key] = str(csv_path)

    summary_path = out_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(_to_serializable(results_dict), f, indent=2)

    artifact_paths["summary_json"] = str(summary_path)
    return artifact_paths
#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------

def train_xgboost(config_path="XGtrain.yaml"):
    cfg = load_config(config_path)

    # =============================
    # 1) Load
    # =============================
    df = pd.read_csv(cfg["data"]["path"]).copy()

    # =============================
    # 2) Preprocess
    # =============================
    if cfg["preprocessing"]["encode_product_id"]:
        df["Product_ID"] = df["Product ID"].str[0]
        enc = LabelEncoder()
        df["Product_ID"] = enc.fit_transform(df["Product_ID"])

    X_raw = df[cfg["features"]["input"]]
    X = clean_feature_names(X_raw)
    feature_name_mapping = dict(zip(X.columns, X_raw.columns))
    y = df[cfg["features"]["target"]].astype(int)

    # =============================
    # 3) Stratified splits
    # =============================
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        stratify=y,
        random_state=cfg["data"]["random_state"]
    )

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp,
        test_size=cfg["split"]["cal_size"],  # 0.25 of 0.8 = 0.2 overall
        stratify=y_temp,
        random_state=cfg["data"]["random_state"]
    )

    # =============================
    # 4) Handle imbalance
    # =============================
    if cfg["model"]["scale_pos_weight"] is None:
        # (neg / pos)
        pos = y_train.sum()
        neg = len(y_train) - pos
        #scale_pos_weight = (neg / pos) if pos > 0 else 1.0
        scale_pos_weight = 1.5 * (neg / pos) if pos > 0 else 1.0
    else:
        scale_pos_weight = cfg["model"]["scale_pos_weight"]

    # =============================
    # 5) Train base model
    # =============================
    base_model = XGBClassifier(
        n_estimators=cfg["model"]["n_estimators"],
        max_depth=cfg["model"]["max_depth"],
        learning_rate=cfg["model"]["learning_rate"],
        subsample=cfg["model"]["subsample"],
        colsample_bytree=cfg["model"]["colsample_bytree"],
        reg_lambda=cfg["model"]["reg_lambda"],
        random_state=cfg["model"]["random_state"],
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        eval_metric="logloss",
        use_label_encoder=False
    )

    base_model.fit(X_train, y_train)

    # =============================
    # 6) Calibration
    # =============================
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method=cfg["calibration"]["method"],
        cv="prefit"
    )
    calibrated_model.fit(X_cal, y_cal)

    # =============================
    # 7) Threshold search (on calibration)
    # =============================
    from sklearn.metrics import recall_score

    cal_proba = calibrated_model.predict_proba(X_cal)[:, 1]

    tmin = cfg["threshold"]["search_min"]
    tmax = cfg["threshold"]["search_max"]
    steps = cfg["threshold"]["steps"]

    thresholds = np.linspace(tmin, tmax, steps)

    from sklearn.metrics import recall_score, precision_score

    TARGET_RECALL = cfg["threshold"].get("target_recall", 0.90)

    best_threshold = None
    best_precision = -1

    for t in thresholds:
        preds = (cal_proba >= t).astype(int)

        recall = recall_score(y_cal, preds)
        precision = precision_score(y_cal, preds, zero_division=0)

        # Keep only thresholds that satisfy recall requirement
        if recall >= TARGET_RECALL:
            if precision > best_precision:
                best_precision = precision
                best_threshold = t

    # 🔴 SAFETY FALLBACK (important)
    if best_threshold is None:
        print("WARNING: No threshold meets target recall. Falling back to max recall.")

        best_threshold = thresholds[0]
        best_score = -1

        for t in thresholds:
            preds = (cal_proba >= t).astype(int)
            recall = recall_score(y_cal, preds)

            if recall > best_score:
                best_score = recall
                best_threshold = t

    # DEBUG OUTPUT
    print(f"Chosen Threshold: {best_threshold}")
    print(f"Calibration Recall: {recall_score(y_cal, (cal_proba >= best_threshold)):.4f}")
    print(f"Calibration Precision: {precision_score(y_cal, (cal_proba >= best_threshold), zero_division=0):.4f}")

    # =============================
    # 8. Step 5: Certainty + calibration tracking
    # =============================
    #probs_cal = calibrated_model.predict_proba(X_cal) if "calibrated_model" in locals() else calibrated_rf.predict_proba(X_cal)
    probs_cal = calibrated_model.predict_proba(X_cal)
    certainty_cal = probs_cal.max(axis=1)

    step5_calibration = {
        "brier_score": float(brier_score_loss(y_cal, probs_cal[:, 1])),
        "log_loss": float(log_loss(y_cal, probs_cal[:, 1], labels=[0, 1])),
        "certainty_mean": float(np.mean(certainty_cal)),
        "certainty_median": float(np.median(certainty_cal)),
        "certainty_min": float(np.min(certainty_cal)),
        "certainty_max": float(np.max(certainty_cal))
    }

    # =============================
    # 9. Step 6: Split conformal prediction
    # =============================
    alpha = cfg["conformal"]["alpha"]

    conformal_info = compute_conformal_quantile(
        probs_cal=probs_cal,
        y_cal=y_cal,
        alpha=alpha
    )

    cp_sets_cal, cp_sizes_cal, cp_amb_cal = build_conformal_sets(
        probs_cal,
        conformal_info["q_alpha"]
    )

    cp_summary_cal = summarize_conformal(cp_sets_cal, y_cal)
    step6_conformal_cal = {
        "alpha": float(alpha),
        "q_level": float(conformal_info["q_level"]),
        "q_alpha": float(conformal_info["q_alpha"]),
        "coverage_cal": float(cp_summary_cal["coverage"]),
        "avg_set_size_cal": float(cp_summary_cal["avg_set_size"]),
        "singleton_rate_cal": float(cp_summary_cal["singleton_rate"]),
        "ambiguous_rate_cal": float(cp_summary_cal["ambiguous_rate"]),
        "empty_set_rate_cal": float(cp_summary_cal["empty_set_rate"])
    }

    # =============================
    # 10. Step 7: Cost-aware deferral threshold on calibration set
    # =============================
    tau_values = np.linspace(
        cfg["deferral"]["tau_min"],
        cfg["deferral"]["tau_max"],
        cfg["deferral"]["steps"]
    )

    deferral_search_df, best_deferral = tune_deferral_threshold(
        probs_cal=probs_cal,
        y_cal=y_cal,
        base_threshold=best_threshold,
        tau_values=tau_values,
        c_fn=cfg["deferral"]["c_fn"],
        c_fp=cfg["deferral"]["c_fp"],
        c_def=cfg["deferral"]["c_def"],
        cp_sets_cal=cp_sets_cal,
        defer_on_ambiguous=cfg["conformal"]["defer_on_ambiguous"]
    )

    step7_deferral = {
        "best_tau": float(best_deferral["tau"]),
        "best_cost_per_sample": float(best_deferral["cost_per_sample"]),
        "best_total_cost": float(best_deferral["total_cost"]),
        "coverage_at_best_tau": float(best_deferral["coverage"]),
        "deferred_fraction_at_best_tau": float(best_deferral["deferred_fraction"]),
        "retained_error_at_best_tau": (
            None if pd.isna(best_deferral["retained_error"]) else float(best_deferral["retained_error"])
        ),
        "retained_recall_at_best_tau": (
            None if pd.isna(best_deferral["retained_recall"]) else float(best_deferral["retained_recall"])
        ),
        "retained_precision_at_best_tau": (
            None if pd.isna(best_deferral["retained_precision"]) else float(best_deferral["retained_precision"])
        )
    }

    print(f"Best deferral tau: {step7_deferral['best_tau']:.4f}")
    print(f"Deferral cost/sample: {step7_deferral['best_cost_per_sample']:.4f}")
    print(f"Coverage at best tau: {step7_deferral['coverage_at_best_tau']:.4f}")

    # =============================
    # 11. Step 8 + Step 9: Final held-out test evaluation
    # =============================
    #probs_test = calibrated_model.predict_proba(X_test) if "calibrated_model" in locals() else calibrated_rf.predict_proba(X_test)
    probs_test = calibrated_model.predict_proba(X_test)
    certainty_test = probs_test.max(axis=1)

    cp_sets_test, cp_sizes_test, cp_amb_test = build_conformal_sets(
        probs_test,
        conformal_info["q_alpha"]
    )

    cp_summary_test = summarize_conformal(cp_sets_test, y_test)

    final_policy = evaluate_deferral_policy(
        probs=probs_test,
        y_true=y_test,
        base_threshold=best_threshold,
        tau=step7_deferral["best_tau"],
        c_fn=cfg["deferral"]["c_fn"],
        c_fp=cfg["deferral"]["c_fp"],
        c_def=cfg["deferral"]["c_def"],
        cp_sets=cp_sets_test,
        defer_on_ambiguous=cfg["conformal"]["defer_on_ambiguous"]
    )

    y_pred_test = (probs_test[:, 1] >= best_threshold).astype(int)
    y_proba_test = probs_test[:, 1]

    retained_mask_test = final_policy["retained_mask"].astype(bool)

    if retained_mask_test.sum() > 0:
        retained_cm = confusion_matrix(y_test[retained_mask_test], y_pred_test[retained_mask_test])
        retained_report = classification_report(
            y_test[retained_mask_test],
            y_pred_test[retained_mask_test],
            output_dict=True,
            zero_division=0
        )
    else:
        retained_cm = np.array([[0, 0], [0, 0]])
        retained_report = {}

    step9_final_eval = {
        "overall_test": {
            "confusion_matrix": confusion_matrix(y_test, y_pred_test),
            "classification_report": classification_report(y_test, y_pred_test, output_dict=True, zero_division=0),
            "roc_auc": float(roc_auc_score(y_test, y_proba_test)),
            "pr_auc": float(average_precision_score(y_test, y_proba_test)),
            "base_threshold": float(best_threshold)
        },
        "conformal_test": {
            "coverage_test": float(cp_summary_test["coverage"]),
            "avg_set_size_test": float(cp_summary_test["avg_set_size"]),
            "singleton_rate_test": float(cp_summary_test["singleton_rate"]),
            "ambiguous_rate_test": float(cp_summary_test["ambiguous_rate"]),
            "empty_set_rate_test": float(cp_summary_test["empty_set_rate"])
        },
        "selective_prediction_test": {
            "tau": float(final_policy["tau"]),
            "coverage": float(final_policy["coverage"]),
            "deferred_fraction": float(final_policy["deferred_fraction"]),
            "deferred_count": int(final_policy["deferred_count"]),
            "retained_error": (
                None if pd.isna(final_policy["retained_error"]) else float(final_policy["retained_error"])
            ),
            "retained_recall": (
                None if pd.isna(final_policy["retained_recall"]) else float(final_policy["retained_recall"])
            ),
            "retained_precision": (
                None if pd.isna(final_policy["retained_precision"]) else float(final_policy["retained_precision"])
            ),
            "fn": int(final_policy["fn"]),
            "fp": int(final_policy["fp"]),
            "tp": int(final_policy["tp"]),
            "tn": int(final_policy["tn"]),
            "total_cost": float(final_policy["total_cost"]),
            "cost_per_sample": float(final_policy["cost_per_sample"]),
            "retained_confusion_matrix": retained_cm,
            "retained_classification_report": retained_report
        }
    }

    # =============================
    # 12. Save step-by-step artifacts
    # =============================
    cal_df = pd.DataFrame({
        "y_true": np.asarray(y_cal).astype(int),
        "p0": probs_cal[:, 0],
        "p1": probs_cal[:, 1],
        "certainty": certainty_cal,
        "cp_set": [cp_set_to_str(s) for s in cp_sets_cal],
        "cp_set_size": cp_sizes_cal,
        "cp_ambiguous": cp_amb_cal,
        "cp_covered": cp_summary_cal["covered_vector"]
    })

    test_df = pd.DataFrame({
        "y_true": np.asarray(y_test).astype(int),
        "p0": probs_test[:, 0],
        "p1": probs_test[:, 1],
        "certainty": certainty_test,
        "base_pred": y_pred_test,
        "final_pred_with_deferral": final_policy["pred_with_deferral"],
        "deferred": final_policy["defer_mask"],
        "retained": final_policy["retained_mask"],
        "cp_set": [cp_set_to_str(s) for s in cp_sets_test],
        "cp_set_size": cp_sizes_test,
        "cp_ambiguous": cp_amb_test,
        "cp_covered": cp_summary_test["covered_vector"]
    })

    results = {
        "model": "XGBoost",

        "step4_thresholding": {
            "base_threshold": float(best_threshold),
            "target_recall": float(cfg["threshold"]["target_recall"])
        },
        "step5_calibration": step5_calibration,
        "step6_conformal_calibration": step6_conformal_cal,
        "step7_deferral_tuning": step7_deferral,
        "step9_final_evaluation": step9_final_eval
    }

    if cfg["output"].get("save_step_artifacts", True):
        artifact_paths = save_pipeline_artifacts(
            cfg=cfg,
            model_name=cfg["output"]["model_name"],
            results_dict=results,
            dfs_dict={
                "step5_step6_calibration_records": cal_df,
                "step7_deferral_search": deferral_search_df,
                "step9_test_predictions": test_df
            }
        )
        results["artifact_paths"] = artifact_paths

    # =============================
    # 9) Save calibrated model
    # =============================
    if cfg["output"]["save_model"]:
        os.makedirs(os.path.dirname(cfg["output"]["model_path"]), exist_ok=True)
        joblib.dump(calibrated_model, cfg["output"]["model_path"])

    return base_model, calibrated_model, X_test, y_test, y_pred_test, y_proba_test, results, cfg, feature_name_mapping