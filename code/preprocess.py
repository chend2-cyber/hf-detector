# -*- coding: utf-8 -*-
"""
preprocess.py — 数据预处理
流程：WFDB 读取 → 250Hz 重采样到 128Hz → 10s 切段(1280点) → 逐段 z-score
      → 按病人(记录) 7:1.5:1.5 划分 → 打乱 → 保存 .npy 到 data/processed
运行：python code/preprocess.py
"""
import re
from math import gcd

import numpy as np
import wfdb
from scipy.signal import resample_poly

from config import (DATA_RAW, DATA_PROCESSED, TARGET_FS, SEGMENT_LEN,
                    TRAIN_RATIO, VAL_RATIO, SEED, set_seed)

MAX_SEG_PER_PATIENT = 200  # 每病人最多 200 段：nsrdb 单条 24h 有 ~9000 段，不设上限会淹没数据集


def load_ecg(hea_path):
    """读一条记录的 0 号通道（两个库的 0 号通道都是 ECG 导联）"""
    rec = wfdb.rdrecord(str(hea_path.with_suffix("")))
    return rec.p_signal[:, 0].astype(np.float64), rec.fs


def resample_to_128(sig, fs):
    """多相滤波重采样，质量优于线性插值；128/250 约分后 = 64/125"""
    if fs == TARGET_FS:
        return sig
    g = gcd(TARGET_FS, fs)
    return resample_poly(sig, up=TARGET_FS // g, down=fs // g)


def make_segments(sig):
    """不重叠切段 + 逐段 z-score；含 NaN 或近乎平直的坏段直接丢弃"""
    segs = []
    for i in range(0, len(sig) - SEGMENT_LEN + 1, SEGMENT_LEN):
        seg = sig[i:i + SEGMENT_LEN]
        if np.isnan(seg).any():
            continue
        std = seg.std()
        if std < 1e-8:
            continue
        segs.append((seg - seg.mean()) / std)
        if len(segs) >= MAX_SEG_PER_PATIENT:
            break
    return segs


def split_records(records, rng):
    """按记录(=病人)划分 7:1.5:1.5 —— 同一病人的段绝不跨集合，防数据泄漏"""
    rng.shuffle(records)
    n = len(records)
    n_tr = int(round(n * TRAIN_RATIO))
    n_va = int(round(n * VAL_RATIO))
    return records[:n_tr], records[n_tr:n_tr + n_va], records[n_tr + n_va:]


def main():
    rng = np.random.default_rng(SEED)
    set_seed()
    buckets = {"train": [], "val": [], "test": []}  # 每项 = (段, 标签)

    # 每个类别内部各自划分，保证三个集合里两类都有
    for sub, label in (("chfdb", 1), ("nsrdb", 0)):
        heas = sorted((DATA_RAW / sub).glob("*.hea"))
        # 只留 chf01 / 16265 这种正式记录，排除 chf01.hea- 等历史版本文件
        heas = [p for p in heas if re.fullmatch(r"chf\d{2}|\d{5}", p.stem)]
        if not heas:
            raise FileNotFoundError(f"{DATA_RAW / sub} 下没找到记录，检查数据是否解压到位")
        print(f"\n== {sub}: {len(heas)} 个病人, 标签={label} ==")
        tr, va, te = split_records(heas, rng)
        for split, recs in (("train", tr), ("val", va), ("test", te)):
            for p in recs:
                sig, fs = load_ecg(p)
                sig = resample_to_128(sig, fs)
                segs = make_segments(sig)
                for seg in segs:
                    buckets[split].append((seg, label))
                print(f"  [{split:5s}] {p.stem:8s} fs={fs}  切出 {len(segs)} 段")

    print()
    for split, items in buckets.items():
        rng.shuffle(items)
        X = np.stack([s for s, _ in items]).astype(np.float32)
        y = np.array([l for _, l in items], dtype=np.float32)
        np.save(DATA_PROCESSED / f"X_{split}.npy", X)
        np.save(DATA_PROCESSED / f"y_{split}.npy", y)
        print(f"{split:5s}: X{X.shape}  心衰占比={y.mean():.1%}  -> {DATA_PROCESSED}")
    print("\n预处理完成 ✔")


if __name__ == "__main__":
    main()