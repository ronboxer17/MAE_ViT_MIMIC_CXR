from abc import abstractmethod, ABC
from typing import List


class BaseModel(ABC):
    def __init__(self, labels: List[str], device='cpu'):
        self.labels = labels
        self.device = device

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError('forward method must be implemented')
