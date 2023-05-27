from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from final_project.datamodels import ConfigModelForTraining, EpochResult


class ModelTrainer:
    def __init__(
        self,
        model_for_training: ConfigModelForTraining,
        train_ratio: float = 0.8,
        batch_size: int = 16,
        num_epochs: int = 10,
        device: str = "cpu",
    ):
        # unpack the model_for_training
        self.model = model_for_training.model
        self.criterion = model_for_training.criterion
        self.optimizer = model_for_training.optimizer
        self.scheduler = model_for_training.scheduler
        self.num_classes = self.model.possible_labels

        self.train_ratio = train_ratio
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        self.device = device
        self.model.to(self.device)

        self.train_dataloader = self.create_dataloaders(
            model_for_training.train_dataset, shuffle=True
        )
        self.val_dataloader = self.create_dataloaders(model_for_training.val_dataset)
        self.test_dataloader = self.create_dataloaders(model_for_training.test_dataset)

    def create_dataloaders(
        self, data: datasets.ImageFolder, shuffle: bool = False
    ) -> Optional[DataLoader]:
        # Load the dataset using ImageFolder
        if not data:
            return None
        return DataLoader(data, batch_size=self.batch_size, shuffle=shuffle)

    def train(self):
        self.model.train()

        print(f"Start Training on {self.device}")
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            running_loss = 0
            # TODO: Add tqdm here

            # Training loop
            for i, (inputs, labels) in enumerate(self.train_dataloader):
                if i > 1:
                    continue

                batch_loss = self.train_batch(inputs, labels)

                running_loss += batch_loss

            epoch_loss = running_loss / len(self.train_dataloader)
            print(
                f"Epoch {epoch + 1}/{self.num_epochs} - Training Loss: {epoch_loss:.4f}"
            )

            # Validation loop
            self.validate_epoch(epoch + 1, self.num_epochs)

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

        print(f"Batch Loss: {loss.item():.4f}")
        return loss.item()

    def validate_epoch(self, current_epoch, total_epochs) -> EpochResult:
        self.model.eval()

        all_losses, all_labels, all_predictions = [], [], []

        with torch.no_grad():
            for epoch, (inputs, labels) in enumerate(self.val_dataloader):
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
        print(
            f"Epoch {current_epoch}/{total_epochs} - Last loss: {epoch_result.losses[-1]:.4f}",
            f"Epoch accuracy: {epoch_result.accuracy:.4f}",
            f"Epoch precision: {epoch_result.precision:.4f}",
            f"Epoch recall: {epoch_result.recall:.4f}",
            f"Epoch f1: {epoch_result.f1:.4f}",
        )
        return epoch_result

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)


if __name__ == "__main__":
    from final_project.datamodels import ConfigModelForTraining
    from final_project.datasets.parking_dataset import load_parking_dataset
    from final_project.models.mae import MAE
    from final_project.models.resnet import ResNet

    # TRAIN RESNET
    def train_resnet():
        train_ratio = 0.8

        dataset = load_parking_dataset()
        train_size = int(train_ratio * len(dataset))

        # Split the dataset into training and validation sets
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, len(dataset) - train_size]
        )
        model = ResNet(possible_labels=dataset.classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        model_for_training = ConfigModelForTraining(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        trainer = ModelTrainer(model_for_training, num_epochs=1)
        trainer.train()

    def train_mae():
        train_ratio = 0.8
        # feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/vit-mae-base')
        # transformer = lambda x: feature_extractor(images=x, return_tensors="pt")
        dataset = load_parking_dataset()
        train_size = int(train_ratio * len(dataset))

        # Split the dataset into training and validation sets
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, len(dataset) - train_size]
        )
        model = MAE(possible_labels=dataset.classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        model_for_training = ConfigModelForTraining(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        trainer = ModelTrainer(model_for_training, num_epochs=1)
        trainer.train()

    def train_mae_with_mimic():
        from final_project.datasets.mimic_dataset import load_mimic_dataset
        train_ratio = 0.8

        dataset = load_mimic_dataset()

        train_size = int(train_ratio * len(dataset))

        # Split the dataset into training and validation sets
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, len(dataset) - train_size]
        )
        # todo: does dataset has dataset.classes?
        model = MAE(possible_labels=dataset.classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        model_for_training = ConfigModelForTraining(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        trainer = ModelTrainer(model_for_training, num_epochs=1)
        trainer.train()
