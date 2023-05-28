from final_project.datasets.mimic import build_dataset
from final_project.datasets.parking import train_test_mock_data
from final_project.trainers.datamodels import TrainingConfig
from final_project.trainers.trainer import ModelTrainer
from final_project.models.mae import MAE
from final_project.models.resnet import ResNet
import torch

train_dataset, val_dataset = train_test_mock_data()


# define parameters
lr = 1e-6
num_epochs = 1
batch_size = 32
device = "cpu"

model = ResNet()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

training_config = TrainingConfig(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=None,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
)

trainer = ModelTrainer(training_config, num_epochs=num_epochs, batch_size=batch_size, device=device, save_model=True)
trainer.train()

