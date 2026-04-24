import pandas as pd
import numpy as np
import yaml
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score
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
    # =============================
    # 7) Threshold search (on calibration)
    # =============================
    from sklearn.metrics import recall_score

    cal_proba = calibrated_model.predict_proba(X_cal)[:, 1]

    tmin = cfg["threshold"]["search_min"]
    tmax = cfg["threshold"]["search_max"]
    steps = cfg["threshold"]["steps"]

    thresholds = np.linspace(tmin, tmax, steps)
#------------------------------------------------------
    # best_threshold = 0.5
    # best_score = -1

    # for t in thresholds:
    #     preds = (cal_proba >= t).astype(int)
    #     score = recall_score(y_cal, preds)
    #
    #     if score > best_score:
    #         best_score = score
    #         best_threshold = t

    # --- DEBUG / LOGGING ---
#-----------------------------------------------------

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
    #print(f"Calibration Recall: {recall_score(y_cal, (cal_proba >= best_threshold)):.4f}")
    # =============================
    # 8) Final test evaluation
    # =============================
    y_proba_test = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= best_threshold).astype(int)

    results = {
        "confusion_matrix": confusion_matrix(y_test, y_pred_test),
        "classification_report": classification_report(y_test, y_pred_test, output_dict=True),
        "roc_auc": roc_auc_score(y_test, y_proba_test),
        "pr_auc": average_precision_score(y_test, y_proba_test),
        "threshold": best_threshold,
        "target_recall": TARGET_RECALL
    }

    # =============================
    # 9) Save calibrated model
    # =============================
    if cfg["output"]["save_model"]:
        os.makedirs(os.path.dirname(cfg["output"]["model_path"]), exist_ok=True)
        joblib.dump(calibrated_model, cfg["output"]["model_path"])

    return base_model, calibrated_model, X_test, y_test, y_pred_test, y_proba_test, results, cfg, feature_name_mapping