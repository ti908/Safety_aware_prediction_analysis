# ======================================
# Explainable Predictive Maintenance (Decision Tree)
# Replication of Matzka (2020) approach
# ======================================

# Importation of necessary dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

def trainRF():
    df = pd.read_csv('Data_file/ai4i2020.csv')

    df['Product_ID'] = df['Product ID'].str[0]
    encoder = LabelEncoder()
    df['Product_ID'] = encoder.fit_transform(df['Product_ID'])

    X = df[['Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]',
            'Tool wear [min]', 'Product_ID']]
    y = df['Machine failure']

    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight=class_weights,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # 👇 RETURN instead of just printing
    return rf, X_train, X_test, y_train, y_test

