

import numpy as np
import yaml
import joblib
import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import RUSBoostClassifier

def load_config(config_path="RUStrain.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score


def train_rusboost(config_path="RUStrain.yaml"):
    cfg = load_config(config_path)

    # =============================
    # 1. Load Data
    # =============================
    df = pd.read_csv(cfg["data"]["path"])
    df_model = df.copy()

    # =============================
    # 2. Preprocessing
    # =============================
    if cfg["preprocessing"]["encode_product_id"]:
        df_model["Product_ID"] = df_model["Product ID"].str[0]
        encoder = LabelEncoder()
        df_model["Product_ID"] = encoder.fit_transform(df_model["Product_ID"])

    X_raw = df_model[cfg["features"]["input"]]
    y = df_model[cfg["features"]["target"]].astype(int)

    if cfg["preprocessing"]["scale"]:
        scaler = StandardScaler()
        X = pd.DataFrame(
            scaler.fit_transform(X_raw),
            columns=X_raw.columns,
            index=X_raw.index
        )
    else:
        X = X_raw

    # =============================
    # 3. STRATIFIED SPLITS
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
    # 4. Train Model
    # =============================
    base_model = RUSBoostClassifier(
        estimator=DecisionTreeClassifier(
            max_depth=cfg["model"]["estimator"]["max_depth"],
            random_state=cfg["model"]["random_state"]
        ),
        n_estimators=cfg["model"]["n_estimators"],
        learning_rate=cfg["model"]["learning_rate"],
        random_state=cfg["model"]["random_state"]
    )

    base_model.fit(X_train, y_train)

    # =============================
    # 5. Calibration
    # =============================
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method=cfg["calibration"]["method"],
        cv="prefit"
    )
    calibrated_model.fit(X_cal, y_cal)

    # =============================
    # 6. Choose Threshold (on calibration set)
    # =============================
    cal_proba = calibrated_model.predict_proba(X_cal)[:, 1]

    tmin = cfg["threshold"]["search_min"]
    tmax = cfg["threshold"]["search_max"]
    steps = cfg["threshold"]["steps"]
    thresholds = np.linspace(tmin, tmax, steps)

    best_threshold = 0.5
    best_score = -1

    # from sklearn.metrics import recall_score
    #
    # for t in thresholds:
    #     preds = (cal_proba >= t).astype(int)
    #     score = recall_score(y_cal, preds)
    #
    #     if score > best_score:
    #         best_score = score
    #         best_threshold = t
    #
    #     if score > best_score:
    #         best_score = score
    #         best_threshold = t

    from sklearn.metrics import recall_score

    best_threshold = 0.5
    best_score = -1

    for t in thresholds:
        preds = (cal_proba >= t).astype(int)
        score = recall_score(y_cal, preds)

        if score > best_score:
            best_score = score
            best_threshold = t    # --- DEBUG / LOGGING ---
    print(f"Chosen Threshold: {best_threshold}")
    print(f"Calibration Recall: {recall_score(y_cal, (cal_proba >= best_threshold)).astype(float):.4f}")

    # =============================
    # 7. FINAL TEST EVALUATION
    # =============================
    y_proba_test = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= best_threshold).astype(int)

    results = {
        "confusion_matrix": confusion_matrix(y_test, y_pred_test),
        "classification_report": classification_report(y_test, y_pred_test, output_dict=True),
        "roc_auc": roc_auc_score(y_test, y_proba_test),
        "pr_auc": average_precision_score(y_test, y_proba_test),
        "threshold": best_threshold
    }



    # =============================
    # 6. Save model
    # =============================
    if cfg["output"]["save_model"]:
        os.makedirs(os.path.dirname(cfg["output"]["model_path"]), exist_ok=True)
        joblib.dump(calibrated_model, cfg["output"]["model_path"])

    return base_model, calibrated_model, X_test, y_test, y_pred_test, y_proba_test, results


