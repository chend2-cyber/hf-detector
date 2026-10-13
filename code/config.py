# -*- coding: utf-8 -*-
"""全局配置：路径 / 数据参数 / 训练超参 / 模型与设备"""
import random
from pathlib import Path

import numpy as np
import torch

# ============ 1. 路径 ============
BASE_DIR = Path(__file__).resolve().parents[1]  # 项目根 hf_detector/
CODE_DIR = BASE_DIR / "code"
DATA_RAW = BASE_DIR / "data" / "raw"            # 下载的原始数据
DATA_PROCESSED = BASE_DIR / "data" / "processed"  # 切片后的数据
MODEL_DIR = BASE_DIR / "model"                  # 模型权重 .pth
RESULT_DIR = BASE_DIR / "result"                # 指标与图表
APP_DIR = BASE_DIR / "app"

# 关键目录不存在时自动创建，避免 FileNotFoundError
for _d in (DATA_RAW, DATA_PROCESSED, MODEL_DIR, RESULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ============ 2. 数据处理参数 ============
TARGET_FS = 128                              # 统一后的采样率(Hz)
CHF_FS = 250                                 # BIDMC CHF 原始采样率
NSR_FS = 128                                 # MIT-BIH NSR 原始采样率
SEGMENT_SEC = 10                             # 每段心电的秒数
SEGMENT_LEN = TARGET_FS * SEGMENT_SEC        # 1280 点/段（用乘法，不硬编码）

# ============ 3. 训练超参 ============
SEED = 42
TRAIN_RATIO = 0.7                            # 按患者划分
VAL_RATIO = 0.15
TEST_RATIO = 0.15

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10                                # 早停：验证集多少轮没提升就停

# ============ 4. 模型与设备 ============
MODEL_NAMES = ["lstm", "bilstm", "cnn_lstm", "transformer"]
NUM_CLASSES = 1                              # 二分类输出 1 个 logit，配 BCEWithLogitsLoss
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    """固定所有随机源，保证实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    print("=== 路径 ===")
    for _n in ("BASE_DIR", "DATA_RAW", "DATA_PROCESSED", "MODEL_DIR", "RESULT_DIR"):
        print(f"{_n:16s} = {globals()[_n]}")
    print("\n=== 参数 ===")
    print(f"TARGET_FS={TARGET_FS}  SEGMENT_SEC={SEGMENT_SEC}  SEGMENT_LEN={SEGMENT_LEN}")
    print(f"BATCH_SIZE={BATCH_SIZE}  EPOCHS={EPOCHS}  LR={LR}  PATIENCE={PATIENCE}")
    print(f"DEVICE={DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU = {torch.cuda.get_device_name(0)}")