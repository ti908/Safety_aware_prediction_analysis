import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def train_explainable_tree(rf, X_train, X_test, y_train, y_test, max_leaf_nodes=4):
    # ===========================================================
    # 1. CONNECT RF → EXPLAINABLE TREE
    # ===========================================================
    # Feature importance from RF
    rf_importances = pd.Series(rf.feature_importances_, index=X_train.columns)

    # Select top 4 features
    top_features = rf_importances.sort_values(ascending=False).head(4).index.tolist()

    print("Top Features from RF:", top_features)

    # Subset data
    X_train_subset = X_train[top_features]
    X_test_subset = X_test[top_features]

    # Compute class weights again (safe + independent)
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = dict(zip(classes, weights))

    # Train Decision Tree
    dt = DecisionTreeClassifier(
        max_leaf_nodes=max_leaf_nodes,
        class_weight=class_weights,
        random_state=42
    )
    dt.fit(X_train_subset, y_train)

    # Predictions
    y_pred_dt = dt.predict(X_test_subset)

    return dt, top_features, y_pred_dt

def evaluate_tree(y_test, y_pred_dt):
    print('\n--- Simple Decision Tree Classifier ---')
    print(confusion_matrix(y_test, y_pred_dt))
    print(classification_report(y_test, y_pred_dt, digits=4))


def plot_decision_tree(dt, feature_names):
    plt.figure(figsize=(14, 8))
    plot_tree(
        dt,
        feature_names=feature_names,
        class_names=['Normal', 'Failure'],
        filled=True,
        fontsize=10
    )
    plt.title('Explainable Decision Tree for Predictive Maintenance')
    plt.show()