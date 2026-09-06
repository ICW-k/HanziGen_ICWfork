#!/bin/bash
# ============================================================
# 云平台训练脚本（aggressive 档）：尽可能压榨实例性能
# 适配：Cloud Studio / Colab / ModelScope 等云端 GPU 实例
# 特点：workers 留 1 核、prefetch 拉满、显存预留比例更高（0.95）
# 本地机请改用 scripts/train_ldm_local.sh（conservative 档）
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# =============== 基础参数（按需修改） ===============
TARGET_FONT_PATH="fonts/myfont.ttf"   # 目标字体路径（用于数据集构建/分析）
TRAIN_SPLIT_RATIO=0.8                  # 训练集占比（与验证占比相加应为 1.0）
VAL_SPLIT_RATIO=0.2                    # 验证集占比
RANDOM_SEED=9999                       # 随机种子（保证可复现）
BATCH_SIZE=auto                        # 批次样本数：auto=latent 空间默认 128（显存不敏感）
                                       #   手动指定：填入整数，跳过自动推算
NUM_WORKERS=auto                       # DataLoader 并行加载进程数：auto=按 CPU 核数推算（云端留 1 核）
                                       #   手动指定：填入整数，跳过自动推算
PRESET=aggressive                      # 硬件档位：aggressive=云端压榨 / conservative=本地稳妥
LEARNING_RATE=5e-4                     # 初始学习率，实际学习率会根据余弦退火策略动态调整
NUM_EPOCHS=1000                         # 训练轮数
SAMPLE_STEPS=50                        # 样例图生成时的采样步数（用于可视化/评估）
IMG_SAVE_INTERVAL=10                   # （可改，不影响模型质量）可视化对比图保存间隔（单位：epoch），纯肉眼监控
                                       #   不影响的原因：画图在 no_grad 下运行、只写文件，不参与训练也不参与选模型
                                       #   提速可调大（如 100），几乎零代价
LPIPS_EVAL_INTERVAL=10                 # （会影响 best 检查点选择，调大需权衡）LPIPS 评估间隔（单位：epoch）
                                       #   每次评估含 gen/gt 图生成 + PSNR/SSIM/LPIPS 三件套（约 2.5 分钟）
                                       #   LPIPS 决定 best 检查点的选择：间隔越大选点越粗，可能错过质量峰值
                                       #   提速推荐 50（评估开销占比从约 40% 降至 <5%）；不建议超过 100
VAL_EVERY=5                            # （可改，不影响模型质量）验证频率：每隔多少 epoch 跑一次全量验证（1=每epoch）
                                       #   不影响的原因：验证在 no_grad 下不更新参数；LDM 的 best 模型只看 LPIPS，
                                       #   学习率按 epoch 固定步进（scheduler.step()），均不依赖 val loss
                                       #   调大（如 10）的唯一损失：无法及早从 val loss 曲线发现过拟合
                                       #   （曲线看 TensorBoard：runs/LDM/；断点续训由 ckpt_save_interval 兜底）
EVAL_BATCH_SIZE=auto                   # 评估批大小：auto=按显存自适应（>=24G→16，否则→8）
                                       #   手动指定：填入整数，跳过自动推算
DEVICE="cuda"                          # 训练设备：cuda / cpu / mps
                                       #（cuda就是使用Nvidia GPU   mps就是使用Apple Silicon GPU）

# =============== 可选：断点续训 / 混合精度 ===============
RESUME_FROM=""                          # 若需断点续训，填入 LDM .pth 权重路径；否则留空
ENABLE_MIXED_PRECISION=1               # 1 表示启用混合精度（需 CUDA），0 表示关闭
                                       #混合精度可以提升训练速度

# =============== 以下参数无需修改 ===============
# 由字体文件名派生输出文件名/目录
TARGET_FONT_NAME=$(basename "$TARGET_FONT_PATH" | sed -E 's/\.(ttf|otf)$//')

# 预训练 VQ-VAE 路径（用于编码/解码；与上次 LDM 训练保持一致）
PRETRAINED_VQVAE_PATH="checkpoints/vqvae_${TARGET_FONT_NAME}.pth"
MODEL_SAVE_PATH="checkpoints/ldm_${TARGET_FONT_NAME}.pth"
SAMPLE_ROOT="samples_${TARGET_FONT_NAME}/"

# 组装参数
ARGS=()
if [ -n "$RESUME_FROM" ]; then
  ARGS+=(--resume_from "$RESUME_FROM")
fi
if [ "$ENABLE_MIXED_PRECISION" -eq 1 ]; then
  ARGS+=(--mixed_precision)
fi
# BATCH_SIZE / NUM_WORKERS / EVAL_BATCH_SIZE 为 auto 时按硬件实时推算；仅显式数字时才传入覆盖
if [ "$BATCH_SIZE" != "auto" ]; then
  ARGS+=(--batch_size "$BATCH_SIZE")
fi
if [ "$NUM_WORKERS" != "auto" ]; then
  ARGS+=(--num_workers "$NUM_WORKERS")
fi
if [ "$EVAL_BATCH_SIZE" != "auto" ]; then
  ARGS+=(--eval_batch_size "$EVAL_BATCH_SIZE")
fi
ARGS+=(--preset "$PRESET")

# =============== 启动训练 ===============
python train_ldm.py \
    --split_ratios "$TRAIN_SPLIT_RATIO" "$VAL_SPLIT_RATIO" \
    --random_seed "$RANDOM_SEED" \
    --learning_rate "$LEARNING_RATE" \
    --num_epochs "$NUM_EPOCHS" \
    --val_every "$VAL_EVERY" \
    --pretrained_vqvae_path "$PRETRAINED_VQVAE_PATH" \
    --model_save_path "$MODEL_SAVE_PATH" \
    --sample_root "$SAMPLE_ROOT" \
    --sample_steps "$SAMPLE_STEPS" \
    --img_save_interval "$IMG_SAVE_INTERVAL" \
    --lpips_eval_interval "$LPIPS_EVAL_INTERVAL" \
    --device "$DEVICE" \
    "${ARGS[@]}"
