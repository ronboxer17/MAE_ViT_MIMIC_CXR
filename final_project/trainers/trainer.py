from typing import Optional, Tuple, Any
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import logging
import datetime
from final_project.trainers.datamodels import TrainingConfig, EpochResult
from final_project.config import LOGS_PATH, MODELS_PATH
from tqdm import tqdm


class ModelTrainer:
    def __init__(
            self,
            _config: TrainingConfig,
            batch_size: int = 16,
            num_epochs: int = 10,
            device: str = "cpu",
            save_model: bool = False,
    ):
        # unpack the _config
        self.model = _config.model
        self.criterion = _config.criterion
        self.optimizer = _config.optimizer
        self.scheduler = _config.scheduler
        self.num_classes = self.model.possible_labels

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.save_model = save_model

        self.output_file_name = self._create_output_file_name()
        self.logger, self.file_handler = self.init_logger()

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        self.device = device
        self.model.to(self.device)

        self.train_dataloader = self.create_dataloaders(
            _config.train_dataset, shuffle=True
        )
        self.val_dataloader = self.create_dataloaders(_config.val_dataset)
        self.test_dataloader = self.create_dataloaders(_config.test_dataset)

    def create_dataloaders(
            self, data: datasets.ImageFolder, shuffle: bool = False
    ) -> Optional[DataLoader]:
        # Load the dataset using ImageFolder
        if not data:
            return None
        return DataLoader(data, batch_size=self.batch_size, shuffle=shuffle)

    def train(self):
        self.model.train()

        self.logger.info(
            f"Start Training model {self.model.model_name} on {self.device},"
            f"{self.num_epochs} epochs, {self.batch_size} batch size"
        )
        best_loss = 3.0
        with tqdm(total=self.num_epochs * self.batch_size, desc='Training Progress', position=0) as pbar:
            for nun_epoch in range(self.num_epochs):
                pbar.set_description(f"Epoch {nun_epoch + 1}/{self.num_epochs}")

                running_loss = 0

                # Training loop
                for num_batch, (inputs, labels) in enumerate(self.train_dataloader):
                    batch_loss = self.train_batch(inputs, labels)
                    if batch_loss < best_loss:
                        best_loss = batch_loss
                    pbar.set_description(
                        f"Epoch: {nun_epoch} Batch {num_batch + 1}/{len(self.train_dataloader)}."
                        f" batch_loss {batch_loss:.4f}. Best Loss so far {best_loss:.4f}"
                    )

                    running_loss += batch_loss

                    epoch_loss = running_loss / len(self.train_dataloader)
                    self.logger.info(
                        f"Epoch {nun_epoch + 1}/{self.num_epochs} - Batch {num_batch} Training Loss: {epoch_loss:.4f}"
                    )
                    pbar.update(1)

                self.logger.info(f'Finished Training Epoch {nun_epoch + 1}')
                # Validation loop
                self.validate_epoch(nun_epoch + 1, self.num_epochs)

                if self.save_model:
                    self.save_model_to_path()

        self.logger.removeHandler(self.file_handler)
        self.file_handler.close()

    def train_batch(self, inputs, labels) -> float:
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(inputs)
        loss = self.criterion(
            outputs, labels
        )  # Access criterion from instance variable

        # Backward pass and optimization
        loss.backward()
        self.optimizer.step()

        # print(f"Batch Loss: {loss.item():.4f}")
        return loss.item()

    def validate_epoch(self, current_epoch, total_epochs) -> EpochResult:
        self.logger.info(f"Epoch {current_epoch}/{total_epochs} - Validating")
        self.model.eval()

        all_losses, all_labels, all_predictions = [], [], []

        with torch.no_grad():
                                           # TODO: make the function work with val dataloader and train dataloader and test dataloader
            for epoch, (inputs, labels) in tqdm(enumerate(self.val_dataloader)):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(
                    outputs, labels
                )  # Access criterion from instance variable
                _, predictions = torch.max(outputs.data, 1)

                # save batch results
                all_losses.append(loss.item())
                all_labels.extend(labels.tolist())
                all_predictions.extend(predictions.tolist())

        epoch_result = EpochResult(
            losses=all_losses,
            labels=all_labels,
            predictions=all_predictions,
        )
        self.logger.info(
            f"Epoch {current_epoch}/{total_epochs} - Last loss: {epoch_result.losses[-1]:.4f}"
            f"Epoch accuracy: {epoch_result.accuracy:.4f}"
            f"Epoch precision: {epoch_result.precision:.4f}"
            f"Epoch recall: {epoch_result.recall:.4f}"
            f"Epoch f1: {epoch_result.f1:.4f}"
        )
        return epoch_result

    def save_model_to_path(self, file_name: Optional[str] = None):
        if not file_name:
            file_name = self.output_file_name

        model_path = os.path.join(MODELS_PATH, f"{file_name}.pth")
        torch.save(self.model.state_dict(), model_path)
        self.logger.info(f"Model saved to {model_path}")

    def init_logger(self) -> Tuple[Any, Any]:
        logger = logging.getLogger("trainer")
        logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(
            os.path.join(LOGS_PATH, f"{self.output_file_name}.log")
        )
        file_handler.setLevel(logging.INFO)

        # Create a formatter and add it to the file handler
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

        # Add the file handler to the logger
        logger.addHandler(file_handler)

        return logger, file_handler

    def _create_output_file_name(self):
        """create a unique file name for the model without extension and path"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{self.model.model_name.replace('/', '-')}_{timestamp}"