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

---

## 5.6 结果不满意怎么排查

### 训练期对比图 `samples_字体/val/` 是什么

训练时每 `IMG_SAVE_INTERVAL` 个 epoch 会存一张**三联图**，文件名 `epoch_XXXX_字符.png`，从左到右三格依次是：

| 位置 | 内容 | 来源 |
|---|---|---|
| 第 1 格 | **ref** | Jigmo 参考字形（模型的输入条件） |
| 第 2 格 | **tgt** | 你的字体对该字的**真实渲染**（标准答案） |
| 第 3 格 | **gen** | LDM 生成的字形（模型输出） |

**关键**：这些字来自 `train.txt` / `val.txt`，也就是**你字体本来就有的字**——不是要补的缺失字。之所以用"已有字"做检验，是因为只有已有字才有真值可以比对；缺失字没有标准答案，无法这样看。

同理 `samples_字体/train/` 是训练集上的三联图（用来看过拟合程度）。

### 怎么看、怎么判断

- **第 2 格（tgt）本身就糊/错位/笔画残缺** → 问题在**数据渲染**，不是模型。检查 `scripts/prepare_dataset.sh` 的 `IMG_WIDTH/IMG_HEIGHT` 与字形居中，重跑 Cell 2
- **第 2 格清楚，但第 3 格（gen）明显糊/缺笔** → 见下方「gen 糊 ≠ 没训够」
- **某些字特别差** → 该字在 Jigmo 与你的字体里风格差异大，属常见现象

### ⚠️ train/val 是不同的字，不能做"同字对比"

训练集与验证集是**互斥划分**（8:2），两边没有任何一个字重合。因此"train 图好、val 图差"指的是**两侧整体水平**的差异：

- `train/` 里 gen ≈ tgt（模型把训练字记住了）
- `val/` 里 gen 明显差 → **过拟合**

注意：这些图是**抽样**——代码只取每个 loader 的第一个 batch（几个字），有偶然性。判断过拟合应以 TensorBoard 的 **train/val loss 剪刀差**为准，图仅作辅助印证。

### gen 糊 ≠ 一定没训够

正确的"同字对比"用法是：**同一个字在不同 epoch 的图**按 epoch 号纵向比较（文件名形如 `epoch_0040_xxxx.png`）：

```bash
ls "samples_字体/val/" | grep <字符码位>      # 按 epoch 号从小到大看演变
```

| 现象 | 结论 | 对策 |
|---|---|---|
| 随 epoch 推进**持续变清晰** | 确实没训够 | 继续训练 |
| 到某 epoch 后**不再改善**（LPIPS 已平） | 轮数不是瓶颈，到模型上限 | 查 VQ-VAE 质量 / 参考字形差异 / 提高 SAMPLE_STEPS |
| train 与 val **都一直糊** | 数据渲染或 VQ-VAE 重建有问题 | 查 `prepare_dataset.sh` 尺寸，或重训 VQ-VAE |

补充：训练期可视化用的是 `train_ldm*.sh` 里的 `SAMPLE_STEPS`（默认 50），而最终推理可用 `inference.sh` 里的 100 步——**训练图糊不代表最终交付的字形糊**，最终质量以 `inference/gen/` 的实际产物为准。

### SAMPLE_STEPS 是什么

扩散模型的**去噪采样步数**（本项目的实现是 DDIM，见 `LDM._synthesize_images_from_references`）：从随机噪声出发，模型一步步去噪生成字形，这个"步数"就是迭代次数。

- 步数越多 → 细节越准、字形错误越少；耗时**线性**增长
- 默认 50；结果不好时**先试 100**（质量明显提升，推理时间翻倍）
- 只想快速预览可设 20（约 2.5 倍速，质量明显下降）
- 注意有两个同名参数：`scripts/inference.sh` 的管**最终补字**（改这个才影响交付质量），`train_ldm*.sh` 的只管训练期可视化/评估图

### 定向排查某个部首 / 某个字（如简体"马"旁）

LDM 的输入是参考字形 `ref`、输出是生成字形 `gen`。某个字错了，**第一步永远是先确认 ref 对不对**——参考字形错了，模型再怎么训练也不可能生成对的结构。

```python
# 1) 先看命名格式（不同版本可能是 uni9A6C / 9A6C / U+9A6C）
import os
d = "samples_字体/inference/ref"
print(os.listdir(d)[:5])
```

```python
# 2) 拼接显示：参考字形 vs 生成字形（把 FONT 换成你的字体名）
import os, re
from PIL import Image
from IPython.display import display

FONT = "你的字体名"                      # 与 FONT_NAME 一致
ref_dir = f"samples_{FONT}/inference/ref"
gen_dir = f"samples_{FONT}/inference/gen"

def code_of(name):                       # 从文件名提取十六进制码位
    m = re.search(r"([0-9a-fA-F]{4,6})", os.path.basename(name))
    return int(m.group(1), 16) if m else None

START, END = 0x9A6C, 0x9BFF               # 马部区间（按需改）
names = [f for f in os.listdir(ref_dir) if f.endswith(".png")
         and (code_of(f) or -1) and START <= code_of(f) <= END]
print("命中", len(names), "个马旁字")

for n in names[:12]:                      # 一次看 12 个
    rp, gp = os.path.join(ref_dir, n), os.path.join(gen_dir, n)
    if not os.path.exists(gp):
        continue
    r, g = Image.open(rp), Image.open(gp)
    w, h = r.width, r.height
    combo = Image.new("L", (w * 2, h))
    combo.paste(r, (0, 0)); combo.paste(g, (w, 0))   # 左=参考 右=生成
    display(combo)
    print(n)
```

**判读**：左边 ref 是标准简体「马」而右边 gen 结构错 → 模型泛化问题；左边 ref 本身就不是简体「马」→ 参考字形选错了。

> 注意：同一字符若被多个 Jigmo 字体同时覆盖，`generation` 按文件名顺序写入，**靠后的字体（jigmo2 / jigmo3）会覆盖靠前的 jigmo.ttf**。如需让常用简繁字只用 jigmo.ttf 作参考，可临时把 jigmo2/3 移出 `fonts/jigmo/`（代价是生僻字失去参考）。

### ⚠️ 缺字无法做 LPIPS 评估，也无法"靶向训练"

LPIPS / PSNR / SSIM 都需要**真值（gt）**做对比：

- 你字体**已有**的字（如繁体「馬」、含馬部的日语字）有真值 → 可以单独挑出来算指标
- **要补的缺失字**（简体「马」）**没有真值** → 既算不了 LPIPS，也没有监督信号可供训练

因此不存在"针对简体马做靶向训练"这条路——模型没有该字在目标字体下的标准答案可学。可行的替代是：确保训练集覆盖目标字体里所有含馬/马部的**已有**字（默认就是全覆盖）、提高 `SAMPLE_STEPS`、继续训练提升整体泛化。

### 排查顺序（按成本从低到高）

1. 把 `scripts/inference.sh` 的 `SAMPLE_STEPS` 改成 100 重跑 Cell 5（零训练成本，常有效）
2. 看 `val/` 三联图定位是数据问题还是模型问题
3. 继续训练 / 重训 LDM（Cell 4）
4. 重训 VQ-VAE（Cell 3，成本最高，但它是重建质量的上限）

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

## 9. 评估指标解读与改进建议

### 9.1 先厘清评估对象

`compute_metrics.sh` 评估的是 `samples_字体/eval_outputs/`（训练验证集，约 676 张），这些是**目标字体已覆盖的字**（有真实 gt 字形），**不是** `inference/gen/` 里那批缺失字的补字结果。因此这组指标衡量"模型生成目标字体风格字形"的能力，间接反映补字质量；真正交付的缺失字（无 gt）无法用指标度量，只能人眼检查。

### 9.2 指标解读

| 指标 | 实测 | 方向 | 解读 |
|---|---|---|---|
| PSNR | 11.9123 | 越高越好，像素级保真 | 偏低，但字形生成任务里 PSNR 天然低（对笔画位置/粗细的像素级差异极敏感 + 生成有随机性），参考价值有限。评估：极差（正常好模型应 20~30+ dB） |
| SSIM | 0.8200 | 越高越好，结构相似度 | 结构相似度良好，字形骨架/布局接近 gt。评估：中等（字形结构大致对，但没到 0.9+） |
| LPIPS | 0.1377 | 越低越好，感知相似度 | 感知距离小，说明"看起来像"。评估：不错（<0.2 算好） |
| FID | 4.9490 | 越低越好，生成分布 vs 真实分布 | <10 已属优秀，生成分布与真实分布高度一致。评估：很好（<10 已算优秀，<5 极佳） |

### 9.3 综合评价

模型**训练收敛良好**：风格迁移与结构保真（SSIM / LPIPS / FID）均达到可用水平。PSNR 低是 LDM 生成式模型的固有现象（不可能逐像素复现），无需纠结。整体属于"可投入补字"的模型水平。

### 9.4 改进建议

1. **不要用 PSNR 作为补字质量的主要判据**，重点盯 LPIPS（越低越好）与 FID（<10 优秀）。
2. 若想再提升质量：
   - 观察训练曲线：LPIPS 仍在下降则可续训（best 检查点已按 LPIPS 保存，续训自动从 `*_last.pth` 恢复进度）。
   - 推理 `SAMPLE_STEPS` 50→100 可提升细节（耗时翻倍）；20→约 2.5 倍速但质量下降。
   - 检查生成结果是否有系统性缺陷（笔画粘连、结构错位），多源于训练集/参考字形问题。
3. **最关键的一步**：这组指标只覆盖"有 gt 的已见字"，真正要交付的是 `inference/gen/` 的缺失字（无 gt、指标测不到）。务必抽样人工目检：
   - 字形结构是否正确（笔画、部件、间架结构）
   - 风格是否与目标字体统一（字重、衬线、笔画粗细）
   - 有无错字、乱笔画、部件缺失
   - 建议从常用字、复杂字、生僻字各抽若干张查看。
4. **推理提速（无损）**：推理在 latent 空间采样，显存占用极低，`BATCH_SIZE` 调大不损失精度（每个字独立采样）。已在 Cell 0 暴露 `INFERENCE_BATCH_SIZE`（默认 64），16G 显存可设 128；DataLoader 的 `num_workers` 会按 CPU 核数自动推算。
