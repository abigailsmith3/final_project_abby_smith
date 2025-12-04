"""Utility helpers for model training and evaluation."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt


def split_data(df: pd.DataFrame):
    """Split dataframe into train/validation/test sets."""
    df_encoded = pd.get_dummies(df, columns=['Contract Length','Gender'], prefix=['Contract Length','Gender'], dtype=int)
    
    X = df_encoded[['Total Spend','Support Calls','Contract Length_Monthly','Last Interaction','Age','Contract Length_Quarterly','Gender_Female']]
    y = df_encoded["Churn"]
    
    X_train, X_validation, y_train, y_validation = train_test_split(X,y, test_size=.3, random_state=123)

    return X_train, X_val, X_test, y_train, y_val, y_test


def plot_roc_curve(y_true, y_score, label: str) -> float:
    """Plot a ROC curve and return the AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {label} (AUC={auc:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return auc


if __name__ == "__main__":
    from src.data.load_data import load_dataset
    from src.data.preprocess import clean_dataset

    raw = load_dataset("data/raw/card_transdata.csv")
    clean = clean_dataset(raw)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(clean)
    print("Train size:", X_train.shape)
    print("Validation size:", X_val.shape)
    print("Test size:", X_test.shape)
