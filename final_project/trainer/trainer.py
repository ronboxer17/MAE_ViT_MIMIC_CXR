import datetime
import json
import logging
import os
from typing import Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from final_project.config import DATASET_TYPES, LOGS_PATH, METRICS_PATH, MODELS_PATH
from final_project.trainer.datamodels import EpochResult, TrainingConfig


class ModelTrainer:
    def __init__(
            self,
            _config: TrainingConfig,
            batch_size: int = 16,
            num_epochs: int = 10,
            device: str = "cpu",
            save_model: bool = False,
    ):
        self._config = _config
        # unpack the _config
        self.model = _config.model
        self.criterion = _config.criterion
        self.optimizer = _config.optimizer
        self.scheduler = _config.scheduler
        self.num_classes = self.model.possible_labels

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.save_model = save_model

        self.output_metrics_file_name = _create_output_metrics_file_name = ()
        self.output_metrics_file_path = self._create_output_metrics_file()
        self.logger, self.file_handler = self.init_logger()

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        self.device = device
        self.model.to(self.device)

        self.train_dataloader = self.create_dataloaders(
            _config.train_dataset, shuffle=True
        )
        self.val_dataloader = self.create_dataloaders(_config.val_dataset, shuffle=True)
        self.test_dataloader = self.create_dataloaders(
            _config.test_dataset, shuffle=True
        )

    def create_dataloaders(
            self, data: Dataset, shuffle: bool = False
    ) -> Optional[DataLoader]:
        # Load the dataset using ImageFolder
        if not data:
            return None
        return DataLoader(data, batch_size=self.batch_size, shuffle=shuffle)

    def train(self):
        self.model.train()

        self.logger.info(f"Start Training model {self.model.model_name} with {self._config.cli_args}")

        with tqdm(
                total=self.num_epochs * len(self.train_dataloader),
                desc="Training Progress",
                position=0,
        ) as pbar:
            for nun_epoch in range(self.num_epochs):
                running_loss = 0

                best_loss = np.Inf
                for num_batch, batch in enumerate(self.train_dataloader):
                    # # TODO: remove this. This is just for debugging
                    # if num_batch > 2:
                    #     break
                    batch_loss = self.train_batch(batch)
                    pbar.set_description(
                        f" Epoch {nun_epoch + 1}/{self.num_epochs}"
                        f" Batch {num_batch + 1}/{len(self.train_dataloader)}"
                        f" Batch_loss {batch_loss:.4f}. Best batch loss so far {best_loss:.4f}"
                    )

                    if batch_loss < best_loss:
                        best_loss = batch_loss
                    running_loss += batch_loss
                    pbar.update(1)

                self.logger.info(
                    f"Finished Training Epoch {nun_epoch + 1} "
                    f"Loss: {running_loss / len(self.train_dataloader):.4f}"
                )

                self.logger.info(f"Epoch {nun_epoch + 1}/{self.num_epochs} - Validating")
                _ = self.validate_epoch(self.val_dataloader, dataset_type="val")

        if self.save_model:
            self.save_model_to_path()

        self.logger.removeHandler(self.file_handler)
        self.file_handler.close()

    def train_batch(self, batch) -> float:
        inputs = batch[0]
        labels = batch[1].to(self.device)
        ids = batch[2] if len(batch) > 2 else []

        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(inputs, device=self.device)
        loss = self.criterion(
            outputs, labels
        )  # Access criterion from instance variable

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate_epoch(self, data_loader: DataLoader, dataset_type: str) -> EpochResult:
        assert dataset_type in DATASET_TYPES, f"{dataset_type} Not in {DATASET_TYPES}"

        self.logger.info(f"Starting Evaluating")
        self.model.eval()

        all_losses, all_labels, all_predictions, all_ids, all_probabilities = (
            [],
            [],
            [],
            [],
            [],
        )
        with torch.no_grad():
            with tqdm(
                    total=len(data_loader),
                    desc=f"Evaluation Progress on- {dataset_type}",
                    position=0,
            ) as pbar:
                for epoch, batch in enumerate(data_loader):
                    inputs = batch[0]
                    labels = batch[1].to(self.device)
                    ids = batch[2] if len(batch) > 2 else []

                    outputs = self.model(inputs, device=self.device)
                    loss = self.criterion(
                        outputs, labels
                    )  # Access criterion from instance variable
                    probabilities = torch.softmax(outputs.data, axis=1)[:, 1].tolist()
                    _, predictions = torch.max(outputs.data, 1)

                    # save batch results
                    all_losses.append(loss.item())
                    all_labels.extend(labels.tolist())
                    all_predictions.extend(predictions.tolist())
                    all_ids.extend(list(ids))
                    all_probabilities.extend(probabilities)
                    pbar.update(1)

        epoch_result = EpochResult(
            losses=all_losses,
            labels=all_labels,
            predictions=all_predictions,
            probabilities=all_probabilities,
            ids=all_ids,
        )
        self.logger.info(
            f"Epoch Last loss: {epoch_result.losses[-1]:.4f} "
            f"Epoch accuracy: {epoch_result.accuracy:.4f} "
            f"Epoch precision: {epoch_result.precision:.4f} "
            f"Epoch recall: {epoch_result.recall:.4f} "
            f"Epoch f1: {epoch_result.f1:.4f} "
            f"Epoch AUC: {epoch_result.roc_auc_score:.4f} "
        )

        self.save_eval_results(epoch_result)

        return epoch_result

    def save_model_to_path(self, file_name: Optional[str] = None):
        if not file_name:
            file_name = self.output_metrics_file_name

        model_path = os.path.join(MODELS_PATH, f"{file_name}.pth")
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")

    def save_eval_results(self, epoch_result: EpochResult) -> None:

        result_dict = epoch_result.dict()
        result_dict.update(
            {
                "accuracy": epoch_result.accuracy,
                "precision": epoch_result.precision,
                "f1": epoch_result.f1,
                "recall": epoch_result.recall,
                "roc_auc_score": epoch_result.roc_auc_score,
            }
        )
        with open(self.output_metrics_file_path, "r") as file:
            results = json.load(file).get("results", {})

        iteration_index = len(results)
        results[str(iteration_index)] = result_dict

        with open(self.output_metrics_file_path, "w") as file:
            json.dump(results, file)

    def init_logger(self) -> Tuple[Any, Any]:
        logger = logging.getLogger("trainer")
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(
            os.path.join(LOGS_PATH, f"{self.output_metrics_file_name}.log")
        )
        file_handler.setLevel(logging.INFO)

        # Create a formatter and add it to the file handler
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        logger.addHandler(file_handler)
        logger.addHandler(logging.StreamHandler())

        return logger, file_handler

    def _create_output_metrics_file_name(self):
        """create a unique file name for the model without extension and path"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.model.model_name.replace('/', '-')}_{timestamp}"

    def _create_output_metrics_file(self) -> str:
        file_name = f"{self.output_metrics_file_name}"
        file_path = os.path.join(METRICS_PATH, f"{file_name}.json")

        with open(file_path, "w") as file:
            json.dump(
                {'metadata': self._config.cli_args, 'results': {}},
                file
            )
        return file_path
