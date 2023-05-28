from typing import Any, List, Optional

from pydantic import BaseModel
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


class TrainingConfig(BaseModel):
    model: Any
    criterion: Any
    optimizer: Any
    scheduler: Optional[Any]
    train_dataset: Optional[Any]
    val_dataset: Optional[Any]
    test_dataset: Optional[Any]


class EpochResult(BaseModel):
    losses: List[float]
    labels: List[int]
    predictions: List[int]

    @property
    def accuracy(self):
        return accuracy_score(self.labels, self.predictions)

    @property
    def f1(self):
        return f1_score(self.labels, self.predictions)

    @property
    def precision(self):
        return precision_score(self.labels, self.predictions)

    @property
    def recall(self):
        return recall_score(self.labels, self.predictions)

    @property
    def confusion_matrix(self):
        return confusion_matrix(self.labels, self.predictions)
