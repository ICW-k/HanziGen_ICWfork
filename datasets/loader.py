from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from configs.ldm_config import LDMDatasetConfig
from configs.vqvae_config import VQVAEDatasetConfig

from .dataset_utils import split_dataset
from .image_dataset import PairedGlyphImageDataset


@dataclass
class TrainValLoader:
    """
    Contains training and validation dataset splits.
    """

    train: DataLoader
    val: DataLoader

    @classmethod
    def from_dataset(
        cls,
        dataset: PairedGlyphImageDataset,
        dataset_config: VQVAEDatasetConfig | LDMDatasetConfig,
        device: torch.device,
    ) -> "TrainValLoader":
        """
        Creates training and validation data loaders from a dataset.
        """
        train_dataset, val_dataset = split_dataset(
            dataset=dataset,
            split_ratios=dataset_config.split_ratios,
            random_seed=dataset_config.random_seed,
        )

        num_workers = dataset_config.num_workers
        pin_memory = device.type == "cuda"
        # persistent_workers + prefetch 显著减少每个 epoch 重建 worker 进程的开销，
        # 且能提前预取下一批数据，避免 GPU 因等待数据而空闲。
        persistent_workers = num_workers > 0

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=dataset_config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=4 if num_workers > 0 else None,
        )

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=dataset_config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=4 if num_workers > 0 else None,
        )

        return cls(train=train_loader, val=val_loader)


@dataclass
class Loader:
    """
    Holds a single data loader for a specific dataset split.
    """

    loader: TrainValLoader

    @classmethod
    def from_dataset_config(
        cls,
        dataset_config: VQVAEDatasetConfig | LDMDatasetConfig,
        device: torch.device,
    ) -> "Loader":
        """
        Creates a single data loader from a dataset directory.
        """
        dataset = PairedGlyphImageDataset(
            target_img_dir=dataset_config.target_img_dir,
            reference_img_dir=dataset_config.reference_img_dir,
        )
        loader = TrainValLoader.from_dataset(
            dataset=dataset,
            dataset_config=dataset_config,
            device=device,
        )

        return cls(loader=loader)
