import torch
import torch.nn as nn


def select_device(
    device: str | torch.device | None = None,
) -> torch.device:
    """
    Select the optimal device for PyTorch operations.
    """
    if device is None:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    if isinstance(device, torch.device):
        device = device
    else:
        device = torch.device(device)

    # 对固定输入尺寸启用 cuDNN benchmark：让 cuDNN 自动挑选最优卷积算法，
    # 显著加速 VQ-VAE / UNet 的前向与反向，且不改变任何数值结果。
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    return device


def print_model_params(model: nn.Module) -> None:
    """
    Print the total number of parameters in a PyTorch model.
    """
    params = sum(p.numel() for p in model.parameters())

    if params >= 1e9:
        formatted_params = f"{params / 1e9:.2f}B"
    elif params >= 1e6:
        formatted_params = f"{params / 1e6:.2f}M"
    elif params >= 1e3:
        formatted_params = f"{params / 1e3:.2f}K"
    else:
        formatted_params = str(params)

    print(f"Total model parameters: {formatted_params}")
