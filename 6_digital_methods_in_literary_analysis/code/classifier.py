"""
SVM: https://arxiv.org/abs/1310.4909
logistic regression: https://ieeexplore.ieee.org/document/8424720
"""

import numpy as np
import pandas as pd
import xgboost
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Classifier:
    def __init__(self) -> None:
        # self.clf = make_pipeline(StandardScaler(), SVC(gamma="auto", random_state=3131))
        self.clf = make_pipeline(StandardScaler(), xgboost.XGBClassifier(device="cuda"))

    def fit(self, train_x: pd.DataFrame, train_y):
        encoded_train_y = LabelEncoder().fit_transform(train_y)
        self.clf.fit(train_x, encoded_train_y)

        # self.clf.fit(train_x, train_y)

    def evaluate(self, eval_x: pd.DataFrame, eval_y):
        predictions = self.clf.predict(eval_x)

        predictions = np.array(list(map(_encoded_to_string, list(predictions))))

        accuracy = metrics.accuracy_score(y_true=eval_y, y_pred=predictions)
        print(f"Accuracy: {accuracy}")
        f1_score = metrics.f1_score(
            y_true=eval_y, y_pred=predictions, average="weighted"
        )
        print(f"F1 score: {f1_score}")
        classification_report = metrics.classification_report(
            y_true=eval_y, y_pred=predictions
        )
        print(f"Classification report: {classification_report}")
        return classification_report

    def get_f1_score(self, eval_x, eval_y):
        predictions = self.clf.predict(eval_x)

        predictions = np.array(list(map(_encoded_to_string, list(predictions))))

        return metrics.f1_score(y_true=eval_y, y_pred=predictions, average="weighted")


def _encoded_to_string(value: int) -> str:
    if value == 0:
        return "austen"
    elif value == 1:
        return "chesterton"
    elif value == 2:
        return "shakespeare"
    else:
        raise ValueError(value)
