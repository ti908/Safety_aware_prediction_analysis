# ======================================
# Explainable Predictive Maintenance (Decision Tree)
# Replication of Matzka (2020) approach
# ======================================

# Importation of necessary dependencies
import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, f1_score

def load_config(config_path="RFtrain.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def trainRF(config_path="RFtrain.yaml"):
    cfg = load_config(config_path)

    # =============================
    # 1. Load Data
    # =============================
    df = pd.read_csv(cfg["data"]["path"])

    # =============================
    # 2. Preprocessing
    # =============================
    if cfg["preprocessing"]["encode_product_id"]:
        df["Product_ID"] = df["Product ID"].str[0]
        encoder = LabelEncoder()
        df["Product_ID"] = encoder.fit_transform(df["Product_ID"])

    X = df[cfg["features"]["input"]]
    y = df[cfg["features"]["target"]]

    # =============================
    # 3. Class Weights
    # =============================
    if cfg["model"]["class_weight"] == "balanced":
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
    else:
        class_weights = None

    # =============================
    # 4. STRATIFIED SPLITS
    # =============================
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        stratify=y,
        random_state=cfg["data"]["random_state"]
    )

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp,
        test_size=cfg["split"]["cal_size"],
        stratify=y_temp,
        random_state=cfg["data"]["random_state"]
    )

    # =============================
    # 5. Train Model
    # =============================
    rf = RandomForestClassifier(
        n_estimators=cfg["model"]["n_estimators"],
        max_depth=cfg["model"]["max_depth"],
        class_weight=class_weights,
        random_state=cfg["model"]["random_state"]
    )
    rf.fit(X_train, y_train)

    # =============================
    # 6. Calibration
    # =============================
    calibrated_rf = CalibratedClassifierCV(
        rf,
        method=cfg["calibration"]["method"],
        cv="prefit"
    )
    calibrated_rf.fit(X_cal, y_cal)

    # =============================
    # 7. Threshold Selection (Calibration Set)
    # =============================
    cal_proba = calibrated_rf.predict_proba(X_cal)[:, 1]

    thresholds = np.linspace(0.01, 0.5, 100)
    best_threshold = 0.5
    best_score = -1


#----------------------------------------
    # from sklearn.metrics import recall_score
    #
    # best_threshold = 0.5
    # best_score = -1
    #
    # for t in thresholds:
    #     preds = (cal_proba >= t).astype(int)
    #     score = recall_score(y_cal, preds)
    #
    #     if score > best_score:
    #         best_score = score
    #         best_threshold = t    # --- DEBUG / LOGGING ---
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

    #print(f"Calibration Recall: {recall_score(y_cal, (cal_proba >= best_threshold)).astype(float):.4f}")

    # =============================
    # 8. FINAL TEST EVALUATION
    # =============================
    y_proba_test = calibrated_rf.predict_proba(X_test)[:, 1]
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
    # 9. Save Model
    # =============================
    if cfg["output"]["save_model"]:
        joblib.dump(calibrated_rf, cfg["output"]["model_path"])

    return rf, calibrated_rf, X_test, y_test, y_pred_test, y_proba_test, results
