import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def train_knn_model(X_train: pd.DataFrame, y_train: pd.Series) -> KNeighborsClassifier:
    """Train and return a 11-NN classifier."""
    knn_model = KNeighborsClassifier(n_neighbors=11)
    knn_model.fit(X_train, y_train)
    return knn_model
