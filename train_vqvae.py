import argparse

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs import VQVAEDatasetConfig, VQVAEModelConfig, VQVAETrainingConfig
from datasets.loader import Loader
from models import VQVAE
from utils.argparse.argparse_utils import update_config_from_args
from utils.hardware.hardware_utils import print_model_params, select_device


def parse_args() -> argparse.Namespace:
    """ """
    parser = argparse.ArgumentParser(description="Train VQVAE model")
    parser.add_argument(
        "--split_ratios", type=float, nargs=2, help="Train/val split ratios"
    )
    parser.add_argument("--random_seed", type=int, help="Random seed")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--num_workers", type=int, help="Number of DataLoader workers")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument(
        "--val_every",
        type=int,
        help="Validate once every N epochs (default 5; 1 = every epoch)",
    )
    parser.add_argument("--model_save_path", type=str, help="Model save path")
    parser.add_argument("--device", type=str, help="Training device (mps, cpu, cuda)")
    parser.add_argument(
        "--resume_from",
        type=str,
        help="Path to VQVAE weights (.pth) to resume from (only loads model state)",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable mixed precision training",
    )

    return parser.parse_args()


def train_vqvae(
    dataset_config: VQVAEDatasetConfig,
    model_config: VQVAEModelConfig,
    training_config: VQVAETrainingConfig,
    device: torch.device,
    resume_from: str | None = None,
):
    """ """
    loader = Loader.from_dataset_config(
        dataset_config=dataset_config,
        device=device,
    )

    vqvae = VQVAE(
        model_config=model_config,
        device=device,
    )

    optimizer = optim.Adam(
        vqvae.parameters(),
        lr=training_config.learning_rate,
    )

    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=training_config.num_epochs,
        eta_min=training_config.min_learning_rate,
    )

    scaler = GradScaler(enabled=training_config.mixed_precision)

    print_model_params(
        model=vqvae,
    )

    # resume_from 由 fit 内部恢复完整状态（model+optimizer+scheduler+epoch）
    vqvae.fit(
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        training_config=training_config,
        resume_from=resume_from,
    )


def main() -> None:
    """ """
    args = parse_args()
    dataset_config = update_config_from_args(
        converting_config=VQVAEDatasetConfig(),
        args=args,
    )
    model_config = update_config_from_args(
        converting_config=VQVAEModelConfig(),
        args=args,
    )
    training_config = update_config_from_args(
        converting_config=VQVAETrainingConfig(),
        args=args,
    )
    device = select_device(args.device)

    train_vqvae(
        dataset_config=dataset_config,
        model_config=model_config,
        training_config=training_config,
        device=device,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
