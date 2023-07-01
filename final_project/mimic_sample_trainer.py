import torch

from final_project.datasets.mimic_sample import build_mimic_sample
from final_project.models.mae import MAE
from final_project.proj_transformers import build_mimic_transform
from final_project.trainer.datamodels import TrainingConfig
from final_project.trainer.trainer import ModelTrainer

# define parameters
lr = 1e-6
num_epochs = 10
batch_size = 128
device = "cuda"

# transformer_model = "facebook/vit-mae-base"
# transformer = AutoImageProcessor.from_pretrained(transformer_model)


train_dataset = build_mimic_sample(build_mimic_transform(is_train=True), is_train=True)
val_dataset = build_mimic_sample(build_mimic_transform(is_train=False), is_train=False)

model = MAE()

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

trainer = ModelTrainer(
    training_config,
    num_epochs=num_epochs,
    batch_size=batch_size,
    device=device,
    save_model=True,
)
trainer.train()
