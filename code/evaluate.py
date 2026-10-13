import json
import csv
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score, f1_score,
                             confusion_matrix, classification_report)

from config import *
from models import build_model
from dataset import get_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    losses, ys, ps = [], [], []
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE).float()
        logit = model(x).squeeze(-1)
        losses.append(criterion(logit, y).item() * x.size(0))
        ps.append(torch.sigmoid(logit).cpu().numpy())
        ys.append(y.cpu().numpy())
    y_true = np.concatenate(ys)
    y_score = np.concatenate(ps)
    y_pred = (y_score >= 0.5).astype(np.int64)
    n = len(y_true)
    return {
        "loss": sum(losses) / n,
        "acc": float((y_pred == y_true).mean()),
        "f1": float(f1_score(y_true, y_pred)),
        "auc": float(roc_auc_score(y_true, y_score)),
        "fpr": roc_curve(y_true, y_score)[0],
        "tpr": roc_curve(y_true, y_score)[1],
        "y_true": y_true,
        "y_pred": y_pred,
    }


def plot_history(models, recs, res):
    sample_keys = list(next(iter(recs.values()))[0].keys())
    prefixes = []
    for k in sample_keys:
        if k != "epoch" and "_" in k:
            p = k.split("_", 1)[0]
            if p not in prefixes:
                prefixes.append(p)
    tr_pref = next((p for p in prefixes if "tr" in p or "train" in p), prefixes[0])
    va_pref = next((p for p in prefixes if "va" in p or "val" in p), prefixes[-1])

    epochs = {m: [d.get("epoch", i + 1) for i, d in enumerate(recs[m])] for m in models}
    va_loss = {m: [d.get(f"{va_pref}_loss") for d in recs[m]] for m in models}
    va_auc = {m: [d.get(f"{va_pref}_auc") for d in recs[m]] for m in models}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, m in enumerate(models):
        axes[0].plot(epochs[m], va_loss[m], color=COLORS[i], label=m)
        axes[1].plot(epochs[m], va_auc[m], color=COLORS[i], label=m)
    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation AUC")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "fig_loss.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    for i, m in enumerate(models):
        ax.plot(res[m]["fpr"], res[m]["tpr"], color=COLORS[i],
                label=f"{m} (AUC={res[m]['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC on Test Set"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "fig_roc.png", dpi=120)
    plt.close(fig)


def main():
    loaders = get_loaders()
    test_loader = loaders["test"]
    res, recs, rows = {}, {}, []
    for name in MODEL_NAMES:
        ckpt = torch.load(MODEL_DIR / f"best_{name}.pt", map_location=DEVICE)
        model = build_model(name).to(DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        r = evaluate_model(model, test_loader)
        res[name], recs[name] = r, json.load(
            open(RESULT_DIR / f"history_{name}.json", encoding="utf-8"))
        rows.append((name, r["loss"], r["acc"], r["f1"], r["auc"]))
        print(f"[{name}] test loss={r['loss']:.4f} acc={r['acc']:.4f} "
              f"f1={r['f1']:.4f} auc={r['auc']:.4f}")
        with open(RESULT_DIR / f"eval_{name}.txt", "w", encoding="utf-8") as f:
            f.write(classification_report(r["y_true"], r["y_pred"], digits=4))
            f.write("\nConfusion matrix [[TN FP][FN TP]]:\n")
            f.write(str(confusion_matrix(r["y_true"], r["y_pred"]).tolist()))

    with open(RESULT_DIR / "eval_summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "test_loss", "test_acc", "test_f1", "test_auc"])
        for name, loss, acc, f1, auc in rows:
            w.writerow([name, f"{loss:.4f}", f"{acc:.4f}", f"{f1:.4f}", f"{auc:.4f}"])

    plot_history(MODEL_NAMES, recs, res)
    print("完成 -> result/ 下: eval_summary.csv / fig_loss.png / fig_roc.png / eval_<model>.txt")

if __name__ == "__main__":
    main()
