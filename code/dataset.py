# -*- coding: utf-8 -*-
"""dataset.py — 把预处理好的 .npy 接到 PyTorch 的 Dataset / DataLoader
运行：python code/dataset.py
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import BATCH_SIZE, DATA_PROCESSED


class ECGDataset(Dataset):
    """输出形状统一为 (1, 1280)：Conv1d 原生要 (C, L)，其它模型在模型内部转置"""

    def __init__(self, split: str):
        X = np.load(DATA_PROCESSED / f"X_{split}.npy")
        y = np.load(DATA_PROCESSED / f"y_{split}.npy")
        assert len(X) == len(y), "X/y 长度不一致，请重跑 preprocess.py"
        # from_numpy 是零拷贝视图，训练时不再反复转换
        self.X = torch.from_numpy(X).unsqueeze(1)   # (N, 1, 1280)
        self.y = torch.from_numpy(y).float()        # (N,)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def get_loaders():
    """返回 {'train','val','test'} 三个 DataLoader"""
    loaders = {}
    for split in ("train", "val", "test"):
        loaders[split] = DataLoader(
            ECGDataset(split),
            batch_size=BATCH_SIZE,
            shuffle=(split == "train"),        # 只有训练集打乱
            num_workers=0,                     # Windows + 数据已在内存，0 最稳最快
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),      # 丢掉不足一个 batch 的尾巴
        )
    return loaders

if __name__ == "__main__":
    loaders = get_loaders()
    for split, dl in loaders.items():
        print(split, len(dl.dataset), len(dl))
    xb, yb = next(iter(loaders["train"]))
    print(xb.shape, yb.shape, xb.dtype, float(xb.mean()), float(xb.std()), float(yb.mean()))
