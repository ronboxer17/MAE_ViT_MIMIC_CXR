import argparse
import torch

from final_project.datasets.mimic_sample import build_mimic_sample
from final_project.models.mae import MAE
from final_project.transformers.proj_transformers import AVAILABLE_TRANSFORMS
from final_project.trainer.datamodels import TrainingConfig
from final_project.trainer.trainer import ModelTrainer
from final_project.models.datamodels import Models
from final_project.models.resnet import ResNet


def main(args):
    # define parameters
    model_name = args.model
    if model_name == Models.MAE_BASE.value:
        model = MAE()
    elif model_name == Models.RESNET18.value:
        model = ResNet()
    else:
        raise ValueError(f"Model {model_name} is not supported")

    transformer = AVAILABLE_TRANSFORMS.get(args.model).get(args.transformer)
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
    parser.add_argument("--model", type=str, default=Models.MAE_BASE.value, choices=[m.value for m in Models],
                        help="Model to train")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--sample_size", type=int, default=10000, help="Sample size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    avb_transforms = [key for model_transforms in AVAILABLE_TRANSFORMS.values() for key in model_transforms.keys()]
    parser.add_argument("--transformer", type=str, default="mae_with_augmentation_prob_025", choices=avb_transforms,
                        help="Choose the transformer you want to train")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training")

    args = parser.parse_args()
    main(args)

#  1st Experiment -  RESNET 18 with Augmentation
# python final_project/experiments_runner.py --model "microsoft/resnet-18" --transformer "resnet_with_augmentation"

#  2st Experiment -  RESNET 18 without Augmentation
# python final_project/experiments_runner.py --model "microsoft/resnet-18" --transformer "resnet_without_augmentation"

#  3st Experiment -  MAE with Augmentation
# python final_project/experiments_runner.py --model "facebook/vit-mae-base" --transformer "mae_with_augmentation_prob_025"

#  4st Experiment -  MAE without Augmentation
# python final_project/experiments_runner.py --model "facebook/vit-mae-base" --transformer "mae_without_augmentation"

#  5st Experiment -  MAE with Augmentation
# python final_project/experiments_runner.py --model "facebook/vit-mae-base" --transformer "mae_with_augmentation_prob_050"

#  6st Experiment -  MAE with Augmentation
# python final_project/experiments_runner.py --model "facebook/vit-mae-base" --transformer "mae_with_augmentation_prob_075"