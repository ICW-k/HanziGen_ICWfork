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
    # DataLoader 预取倍数（每 worker 预取的 batch 数）；aggressive 档拉高以压榨吞吐
    prefetch_factor: int = 4


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

    # 周期保存完整训练状态（模型+optimizer+scheduler+epoch）的间隔，0 表示不周期保存。
    # 周期文件为 model_save_path 同目录下的 *_last.pth；主文件只保留 best
    ckpt_save_interval: int = 5
    # 每隔多少 epoch 跑一次全量验证（1 表示每 epoch 都验证）
    # 验证集不参与梯度更新，5:1 的验证频率可省约 16% 总耗时（train:val = 8:2）
    val_every: int = 5

    tensorboard_log_dir: str = "runs/VQVAE"
    mixed_precision: bool = True
