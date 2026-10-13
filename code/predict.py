import argparse
from pathlib import Path
import numpy as np
import torch

from config import *
from preprocess import load_ecg, resample_to_128, make_segments
from models import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {0: "正常 (Normal)", 1: "心衰 (CHF)"}


def load_model(name):
    ckpt = torch.load(MODEL_DIR / f"best_{name}.pt", map_location=DEVICE, weights_only=True)
    model = build_model(name).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def predict_record(model, rec_path, verbose=True):
    rec_path = Path(rec_path)
    sig, fs = load_ecg(rec_path)
    sig = resample_to_128(sig, fs)
    segs = np.asarray(make_segments(sig), dtype=np.float32)
    if segs.ndim == 1:
        segs = segs.reshape(1, -1)
    if segs.shape[0] == 0:
        print("该记录没有可用片段（全被过滤）")
        return None
    x = torch.from_numpy(segs).unsqueeze(1).to(DEVICE)          # (N, 1, 1280)
    with torch.no_grad():
        probs = torch.sigmoid(model(x).squeeze(-1)).cpu().numpy()
    mean_p = float(probs.mean())
    label = int(mean_p >= 0.5)
    if verbose:
        print(f"记录     : {rec_path}")
        print(f"片段数   : {len(segs)}")
        print(f"片段概率 : min={probs.min():.3f}  max={probs.max():.3f}  mean={mean_p:.3f}")
        print(f"判定     : {LABELS[label]}   (心衰概率 {mean_p:.3f})")
    return mean_p, label


def main():
    ap = argparse.ArgumentParser(description="单条 ECG 记录的心衰预测")
    ap.add_argument("record", help="WFDB 记录路径(不含 .hea/.dat)，例：data/raw/chfdb/chf01")
    ap.add_argument("--model", default="cnn_lstm", choices=MODEL_NAMES)
    args = ap.parse_args()

    model, ckpt = load_model(args.model)
    print(f"模型     : {args.model}  (best_epoch={ckpt['best_epoch']}, "
          f"best_val_auc={ckpt['best_val_auc']:.4f})")
    predict_record(model, args.record)

if __name__ == "__main__":
    main()
