from typing import Any, List, Optional

from pydantic import BaseModel, Field
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)



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
    probabilities: List[float]
    ids: List[str]

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
    def roc_auc_score(self):
        return roc_auc_score(self.labels, self.probabilities)

    @property
    def confusion_matrix(self):
        return confusion_matrix(self.labels, self.predictions)


if __name__ == '__main__':
    losses = [0.0003, 0.031]
    labels = [1, 0]
    predictions = [1, 0]
    probabilities = [0.7, 0.2]
    ids = ['sdfsadf', 'wefasad']
    e = EpochResult(
        losses=losses,
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        ids=ids,
    )
    d = e.dict()
    d.update({
        'accuracy': e.accuracy,
        'precision': e.precision,
        'f1': e.f1,
        'recall': e.recall,
        'roc_auc_score': e.roc_auc_score,
    })
    print(d)
