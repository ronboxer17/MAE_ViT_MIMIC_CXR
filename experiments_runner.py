import argparse
import torch

from final_project.datasets.mimic_sample import build_mimic_sample
from final_project.models.mae import MAE
from final_project.transformers.proj_transformers import AVAILABLE_TRANSFORMS
from final_project.trainer.datamodels import TrainingConfig
from final_project.trainer.trainer import ModelTrainer


def main(args):
    # define parameters
    transformer = AVAILABLE_TRANSFORMS.get(args.transformer)
    transformer_train = transformer.get("train")
    transformer_val = transformer.get("val")

    train_sample_size = args.sample_size
    val_sample_size = min(int(train_sample_size * 0.2), 1830)

    lr = args.lr
    num_epochs = args.epochs
    batch_size = args.batch_size
    device = args.device

    train_dataset = build_mimic_sample(transformer_train, train_sample_size, is_train=True)
    val_dataset = build_mimic_sample(transformer_val, val_sample_size, is_train=False)

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
        cli_args=vars(args)
    )

    trainer = ModelTrainer(
        training_config,
        num_epochs=num_epochs,
        batch_size=batch_size,
        device=device,
        save_model=True,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with specified parameters")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--sample_size", type=int, default=10000, help="Sample size for training")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--transformer", type=str, default="default", choices=AVAILABLE_TRANSFORMS.keys(), help="Choose the transformer you want to train")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training")

    args = parser.parse_args()
    main(args)
