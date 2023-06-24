import torch
from transformers import ViTImageProcessor

from final_project.datasets.mimic import build_mimic_dataset
from final_project.datasets.parking import train_test_mock_data
from final_project.models.mae import MAE
from final_project.models.resnet import ResNet
from final_project.models.convnext import ConvNext
from final_project.trainers.datamodels import TrainingConfig
from final_project.trainers.trainer import ModelTrainer


# define parameters
lr = 1e-6
num_epochs = 1
batch_size = 32
device = "cpu"


def train_mae():
    train_dataset = build_mimic_dataset(is_train=True)
    val_dataset = build_mimic_dataset(is_train=False)
    transform = ViTImageProcessor.from_pretrained('facebook/vit-mae-base')

    train_dataset, val_dataset = train_test_mock_data(transformer=transform)
    model = MAE()
    training_config = TrainingConfig(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=lr),
        scheduler=None,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer = ModelTrainer(
        training_config,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        save_model=True,
    )

    trainer.train()

def train_resnet():
    train_dataset, val_dataset = train_test_mock_data()
    model = ResNet()
    training_config = TrainingConfig(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=lr),
        scheduler=None,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer = ModelTrainer(
        training_config,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        save_model=True,
    )

    trainer.train()


def train_convnext():
    train_dataset, val_dataset = train_test_mock_data()
    model = ConvNext()
    training_config = TrainingConfig(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.AdamW(model.parameters(), lr=lr),
        scheduler=None,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer = ModelTrainer(
        training_config,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        save_model=True,
    )

    trainer.train()

train_mae()