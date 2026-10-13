# -*- coding: utf-8 -*-
"""train.py — 训练模型：早停 + 保存最优权重 + 逐 epoch 历史
用法：
    python code/train.py           # 依次训练全部四个模型
    python code/train.py lstm      # 只训练指定模型（lstm/bilstm/cnn_lstm/transformer）
"""
import json
import sys

import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

from config import (DEVICE, EPOCHS, LR, MODEL_DIR, MODEL_NAMES, PATIENCE,
                    RESULT_DIR, WEIGHT_DECAY, set_seed)
from dataset import get_loaders
from models import build_model


def run_epoch(model, loader, criterion, optimizer=None):
    """跑一个 epoch；optimizer=None 表示只做前向评估（val/test）"""
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss, probs, labels = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb).squeeze(1)              # (B,1) -> (B,)
        loss = criterion(logits, yb)
        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item() * len(yb)
        probs.append(torch.sigmoid(logits).detach().cpu())
        labels.append(yb.detach().cpu())

    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()
    preds = probs > 0.5
    return {
        "loss": total_loss / len(labels),
        "acc": float((preds == labels).mean()),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auc": float(roc_auc_score(labels, probs)),
    }, probs, labels


def train_one(name, loaders):
    print(f"\n===== 训练 {name} =====")
    set_seed()                                     # 四个模型同一起点，保证公平对比
    model = build_model(name)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                          factor=0.5, patience=4)

    best_auc, best_epoch = -1.0, 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        tr, _, _ = run_epoch(model, loaders["train"], criterion, optimizer)
        va, _, _ = run_epoch(model, loaders["val"], criterion)
        scheduler.step(va["auc"])
        history.append({"epoch": epoch,
                        **{f"tr_{k}": v for k, v in tr.items()},
                        **{f"va_{k}": v for k, v in va.items()}})
        print(f"[{name}] ep {epoch:3d} | tr_loss {tr['loss']:.4f} | "
              f"va_loss {va['loss']:.4f} acc {va['acc']:.4f} f1 {va['f1']:.4f} auc {va['auc']:.4f}")

        if va["auc"] > best_auc:                   # 每有提升就落盘，中途断了也不白跑
            best_auc, best_epoch = va["auc"], epoch
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save({"model_name": name, "state_dict": state,
                        "best_epoch": best_epoch, "best_val_auc": best_auc},
                       MODEL_DIR / f"best_{name}.pt")
            print(f"    ^ 保存最优权重 (val_auc {best_auc:.4f})")
        if epoch - best_epoch >= PATIENCE:
            print(f"[{name}] 连续 {PATIENCE} 轮无提升，早停于 epoch {epoch}")
            break

    with open(RESULT_DIR / f"history_{name}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"[{name}] 最优 epoch={best_epoch}  val_auc={best_auc:.4f}")
    torch.cuda.empty_cache()
    return {"model": name, "best_epoch": best_epoch, "val_auc": best_auc}


def main():
    names = sys.argv[1:] or MODEL_NAMES
    bad = [n for n in names if n not in MODEL_NAMES]
    if bad:
        raise SystemExit(f"未知模型 {bad}，可选 {MODEL_NAMES}")
    print(f"设备: {DEVICE} | 待训练模型: {names}")
    loaders = get_loaders()
    rows = [train_one(n, loaders) for n in names]

    print("\n===== 训练汇总 =====")
    for r in sorted(rows, key=lambda x: -x["val_auc"]):
        print(f"{r['model']:12s} best_epoch={r['best_epoch']:3d}  val_auc={r['val_auc']:.4f}")
    with open(RESULT_DIR / "train_summary.csv", "w", encoding="utf-8") as f:
        f.write("model,best_epoch,val_auc\n")
        for r in sorted(rows, key=lambda x: -x["val_auc"]):
            f.write(f"{r['model']},{r['best_epoch']},{r['val_auc']:.4f}\n")

if __name__ == "__main__":
    main()
