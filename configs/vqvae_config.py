from dataclasses import dataclass


@dataclass
class VQVAEDatasetConfig:
    """
    Configuration class for the VQVAE dataset settings.
    """

    target_img_dir: str = "data/target"
    reference_img_dir: str = "data/reference"

    splits_root: str = "charsets"
    split_ratios: tuple[float, float] = (0.8, 0.2)
    random_seed: int = 2025
    batch_size: int = 8
    num_workers: int = 4


@dataclass
class VQVAEModelConfig:
    """
    Configuration class for the VQVAE architecture settings.
    """

    input_img_channels: int = 1
    encoder_base_channels: int = 64
    latent_dim: int = 2
    codebook_size: int = 64
    commitment_cost: float = 0.25


@dataclass
class VQVAETrainingConfig:
    """
    Configuration class for the VQVAE training settings.
    """

    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-6
    num_epochs: int = 100

    model_save_path: str = "checkpoints/vqvae.pth"
    best_model_save_path: str = "checkpoints/vqvae_best.pth"

    # 周期保存完整训练状态（模型+optimizer+scheduler+epoch）的间隔，0 表示不周期保存
    ckpt_save_interval: int = 5
    # 恢复训练起始 epoch（由 resume_from 时自动从检查点读取，通常无需手动设置）
    resume_epoch: int = 0

    tensorboard_log_dir: str = "runs/VQVAE"
    mixed_precision: bool = True
