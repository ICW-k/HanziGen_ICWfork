"""Checkpoint 加载的跨 torch 版本兼容层。"""

import torch


def load_checkpoint(
    path: str,
    map_location="cpu",
):
    """
    加载 checkpoint，兼容新旧 torch 版本与新旧保存格式。

    背景：
    - torch >= 2.6 的 ``torch.load`` 默认 ``weights_only=True``，只接受白名单
      类型（tensor / dict / list / 基本标量等）；checkpoint 里若含白名单外的
      pickle 对象（如完整训练状态中的 scaler 状态等），会抛 ``UnpicklingError``。
    - torch < 1.13 则没有 ``weights_only`` 参数，显式传参会抛 ``TypeError``。

    策略（按顺序探测，任一成功即返回）：
    1. ``weights_only=True``（安全模式，新 torch 默认；纯 state_dict 可直接加载）
    2. 旧 torch 不支持该参数（TypeError）→ 不带参数加载
    3. 安全模式拒绝加载（含非白名单对象）→ 显式 ``weights_only=False`` 完整加载

    Returns:
        checkpoint 内容：可能是纯 state_dict，也可能是含 "model" 键的完整训练状态，
        由调用方自行判别。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # 旧版本 torch 没有 weights_only 参数
        return torch.load(path, map_location=map_location)
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        # 文件不存在 / 无法读取：直接向上抛原始异常，不做二次加载，
        # 避免真实的文件错误被下面的回退逻辑掩盖
        raise
    except Exception:
        # 新版本 torch 安全模式拒绝加载（checkpoint 含非白名单对象），
        # 回退到完整加载。文件来源为本项目自己的训练产物，可信。
        return torch.load(path, map_location=map_location, weights_only=False)
