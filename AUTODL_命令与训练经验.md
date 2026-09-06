# AutoDL 训练命令速查（HanziGen 项目）

> 项目路径示例：`/root/autodl-tmp/HanziGen_ICWfork`
> 数据盘在 `/root/autodl-tmp`（空间大、不计费重置），系统盘 `/root` 空间小——**模型、数据、字体都放数据盘**。

---

## 1. 环境

```bash
# 查看当前 python / 环境
which python && python --version

# 安装项目依赖（缺包报 ModuleNotFoundError 时）
pip install -r requirements.txt

# 安装指定 CUDA 版 PyTorch（一般 AutoDL 镜像已预装，勿随意重装）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 验证 PyTorch 能否看到 GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> 无卡模式调试时 `torch.cuda.is_available()` 返回 False 是正常的，插卡实例上才为 True。

## 2. 硬件查看

```bash
nvidia-smi                # GPU 型号 / 显存占用 / 正在跑的进程（循环刷新加参数 -l 1）
nvidia-smi -l 1           # 每秒刷新，Ctrl+C 退出
htop                      # CPU / 内存占用（q 退出）
df -h                     # 磁盘剩余空间
du -sh data samples_* checkpoints fonts   # 查看项目各目录占用
```

## 3. 代码同步（本地改动 ↔ AutoDL）

### 首次克隆（先开 AutoDL 的 GitHub 加速）

AutoDL 内置学术加速，访问 GitHub 慢或超时的时候先开启：

```bash
source /etc/network_turbo          # 开启加速（对 GitHub / HuggingFace / PyTorch 官方源生效）
unset http_proxy https_proxy       # 不需要加速时关闭
```

```bash
cd /root/autodl-tmp
git clone https://github.com/ICW-k/HanziGen_ICWfork.git
cd HanziGen_ICWfork
```

### 日常同步（本地 push 后在 AutoDL 拉取）

```bash
cd /root/autodl-tmp/HanziGen_ICWfork

# AutoDL 上拉取你 fork 仓库的最新改动（notebook / sh / py 修改后都要拉）
git pull

# 如果在 AutoDL 上手动改过文件（如 sed 改脚本）导致 pull 报冲突，
# 且这些修复已包含在远端仓库里，可以丢弃本地改动后再拉：
git checkout -- . && git pull

# 查看当前代码版本（确认 pull 成功）
git log --oneline -3
```

## 4. 训练

```bash
# 数据准备（无卡实例也能跑）
bash scripts/analyze_font.sh
bash scripts/prepare_dataset.sh
bash scripts/extract_charset.sh

# 训练（有卡实例）
bash scripts/train_vqvae_local.sh     # VQ-VAE
bash scripts/train_ldm_local.sh       # LDM

# 断点续训：脚本里的 RESUME_FROM 已由 notebook Cell 1 自动改写，重跑同一条命令即可
```

**长时间训练建议放后台**（关掉网页/SSH 不中断）：

```bash
# 方式一：tmux（推荐）
tmux new -s train                      # 新建会话
bash scripts/train_ldm_local.sh        # 在里面启动训练
# Ctrl+B 然后按 D  → 脱离会话（训练继续跑）
tmux attach -t train                   # 重新进入查看
tmux ls                                # 列出所有会话

# 方式二：nohup + 日志文件
nohup bash scripts/train_ldm_local.sh > train.log 2>&1 &
tail -f train.log                      # 实时查看日志（Ctrl+C 只退出查看，不影响训练）
```

## 5. 查看 loss 曲线（TensorBoard）

训练数据写在 `runs/LDM/<时间戳>/` 与 `runs/VQVAE/<时间戳>/`。

**logdir 指向"时间戳目录的父级"即可，不用 cd 进 events 文件所在目录。**

以实际路径为例：

```
events 文件: autodl-tmp/HanziGen_ICWfork/runs/LDM/20260906-211725/events.out.tfevents.1788...
                                     ^^^^^^^^ ← logdir 指到这一层（时间戳目录的父级）
```

```bash
cd /root/autodl-tmp/HanziGen_ICWfork
tensorboard --logdir runs/LDM --port 6006 --bind_all
# 想把 VQ-VAE 和 LDM 的曲线一起对比：把 logdir 换成 runs
```

然后在 AutoDL 控制台 →「自定义服务」→ 用 6006 端口打开网页。
（notebook 内也可用 `%load_ext tensorboard` + `%tensorboard --logdir runs/LDM`）

**界面里重点看的曲线**：
- `Loss/train` 与 `Loss/val`：训练/验证损失（val loss 持续回升 + train loss 还在降 = 过拟合信号）
- `Metrics/val/lpips`：LDM 的模型选择指标（越低越好；它创新低时终端会打印 `✅ Best model saved`）

---

## 5.5 训练何时可以提前结束（经验）

**前提**：best 检查点由指标自动保存（主文件永远是最优），提前结束不会损失模型质量，只省机时。

### 两个模型的停止规则

| | VQ-VAE（Cell 3） | LDM（Cell 4） |
|---|---|---|
| best 的选择指标 | **val loss**（重建误差） | **LPIPS**（感知质量） |
| 评估间隔 | `VAL_EVERY=5` | `LPIPS_EVAL_INTERVAL=10` |
| 可结束的信号 | 连续 **3~5 个验证点**（15~25 epoch）val loss 无新低 | 连续 **3~5 个评估点**（30~50 epoch）终端没有再打印 `✅ Best model saved` |
| 参考耗时 | VQ-VAE val loss 变化更大更早收敛 | LPIPS 后期常在小数点后三四位抖动 |

### 实际案例（LDM）

- val loss 从 0.032（301 轮）回升到 0.042（400 轮）——过拟合苗头
- 但 LPIPS 同期 0.1446（360 轮）→ 0.1427（390 轮）仍在创新低
- **结论：两者打架时 LPIPS 说了算**——最终产物是生成字形，LPIPS 才与交付质量直接挂钩；继续练，直到 LPIPS plateau

### 常见误区

- ❌ "LDM 轮数必须 ≥ VQ-VAE 轮数"：两者是不同任务，各以**自己的指标 plateau** 为准。LDM 默认 1000 轮 > VQ-VAE 600 轮，只是因为生成任务更难收敛，不是对齐关系；LDM 完全可能 500 轮就到顶
- ❌ "val loss 回升就要立刻停"：对 LDM 只是观察信号，best 由 LPIPS 锁定
- ❌ "续训时改大 NUM_EPOCHS 延长训练"：余弦退火学习率曲线按总轮数设计，改轮数 = 改变 lr 调度 = 训练动态变化。轮数启动前定好，中途只做"提前停"，不做"延长跑"

## 6. 产出物位置

| 目录 | 内容 |
|---|---|
| `checkpoints/` | `vqvae_字体.pth` / `ldm_字体.pth`（best + 周期检查点，覆盖式） |
| `runs/LDM/`、`runs/VQVAE/` | TensorBoard 日志（loss 曲线） |
| `samples_字体/val/` `train/` | 每 N epoch 的 gen/ref/gt 对比图（肉眼监控） |
| `samples_字体/eval_outputs/` | 评估用图与指标 |
| `samples_字体/inference/gen/` | 最终补字 PNG |
| `svgs_字体/` | SVG 矢量结果 |
| `charsets/` | 缺字表、训练/验证字符集划分 |
| `colab_state.json` | 各阶段完成状态（断连恢复用） |

## 7. 文件传输

```bash
# AutoDL 网盘（控制台 -> 网盘）适合传字体、Jigmo ZIP、VGG 权重等大文件
# scp 直传（本地终端执行）：
scp "本地路径/字体.otf" root@connect.westb.seetacloud.com:/root/autodl-tmp/HanziGen_ICWfork/fonts/

# 打包下载训练结果
cd /root/autodl-tmp && zip -r results.zip HanziGen_ICWfork/checkpoints HanziGen_ICWfork/svgs_*
```

## 8. 常用排查

```bash
ps aux | grep train          # 查看训练进程是否还在
kill -9 <PID>                # 强制结束训练（续训不丢超过 5 epoch 的进度）
nvidia-smi                   # 训练时显存应接近占满；远低于则 batch 可尝试调大

# OOM（显存不足）处理：调小 scripts/train_vqvae_local.sh 的 VRAM_RESERVE_FRACTION（如 0.80）
# 或把 BATCH_SIZE 从 auto 改成具体数字

# 模块找不到：确认 shell 的 python 与 notebook kernel 是同一个环境
which python
python -m pip list | grep -E "rich|potrace|lpips|fonttools"
```
