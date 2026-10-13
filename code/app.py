# -*- coding: utf-8 -*-
"""Flask 心衰检测 Web 应用（固定 cnn_lstm，上传 .hea/.dat，SQLite 存历史）"""
import base64
import io
import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Microsoft YaHei","SimHei"]
plt.rcParams["axes.unicode_minus"]=False
from flask import Flask, request, render_template, redirect, url_for, flash

from config import *
from preprocess import load_ecg, resample_to_128, make_segments
from models import build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = {0: "正常 (Normal)", 1: "心衰 (CHF)"}
MODEL_NAME = "cnn_lstm"                       # 推荐：固定最优模型

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "detections.db"

app = Flask(__name__)
app.secret_key = "hf-detector-dev"            # 仅本地用，换掉即可


def load_default_model():
    ckpt = torch.load(MODEL_DIR / f"best_{MODEL_NAME}.pt",
                      map_location=DEVICE, weights_only=True)
    model = build_model(MODEL_NAME).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


MODEL, CKPT = load_default_model()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, filename TEXT, label TEXT, prob REAL, segs INTEGER)""")
    conn.commit()
    conn.close()


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def plot_wave(sig, n):
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.plot(np.arange(n), sig[:n], color="#1f77b4", lw=0.6)
    ax.set_xlabel("采样点"); ax.set_ylabel("mV")
    ax.set_title("ECG 波形（前 %d 点，约 %.0f s）" % (n, n / TARGET_FS))
    ax.grid(alpha=0.3)
    return fig_to_b64(fig)


def plot_dist(probs):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(probs, bins=20, color="#ff7f0e", edgecolor="white")
    ax.axvline(0.5, color="red", ls="--", lw=1)
    ax.set_xlabel("片段心衰概率"); ax.set_ylabel("片段数")
    ax.set_title("各片段概率分布")
    ax.grid(alpha=0.3)
    return fig_to_b64(fig)


def run_inference(rec_path):
    sig, fs = load_ecg(rec_path)
    sig = resample_to_128(sig, fs)
    segs = make_segments(sig)
    n = len(segs)
    if n == 0:
        raise ValueError("该记录没有可用片段（可能被过滤为空）")
    x = torch.from_numpy(np.stack(segs).astype(np.float32)).unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(MODEL(x).squeeze(-1)).cpu().numpy()
    mean_p = float(probs.mean())
    label = int(mean_p >= 0.5)
    n_view = min(len(sig), 5120)              # 波形只画前 ~40s
    return {
        "mean_p": mean_p, "label": label, "n": n,
        "pmin": float(probs.min()), "pmax": float(probs.max()),
        "wave_png": plot_wave(sig, n_view), "dist_png": plot_dist(probs),
    }


def get_test_auc(name):
    try:
        import csv
        with open(RESULT_DIR / "eval_summary.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["model"] == name:
                    return float(row["test_auc"])
    except Exception:
        pass
    return None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", model=MODEL_NAME,
                           val_auc=CKPT["best_val_auc"],
                           test_auc=get_test_auc(MODEL_NAME))


@app.route("/predict", methods=["POST"])
def predict():
    files = request.files.getlist("files")
    if len(files) < 2:
        flash("请同时上传 .hea 和 .dat 两个文件")
        return redirect(url_for("index"))

    tmp = Path(tempfile.mkdtemp())
    stems = {}
    for f in files:
        name = f.filename or ""
        suffix = Path(name).suffix.lower()
        if suffix in (".hea", ".dat"):
            stem = Path(name).stem
            f.save(str(tmp / (stem + suffix)))
            stems.setdefault(stem, set()).add(suffix)

    rec = None
    for stem, exts in stems.items():
        if {".hea", ".dat"}.issubset(exts):
            rec = tmp / stem
            break
    if rec is None:
        flash("未找到配对的 .hea / .dat 记录")
        return redirect(url_for("index"))

    try:
        result = run_inference(rec)
    except Exception as e:
        flash("预测失败：%s" % e)
        return redirect(url_for("index"))
    finally:
        for p in tmp.glob("*"):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            tmp.rmdir()
        except OSError:
            pass

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO detections (ts, filename, label, prob, segs) VALUES (?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rec.name,
         LABELS[result["label"]], round(result["mean_p"], 4), result["n"]))
    conn.commit()
    conn.close()

    return render_template("result.html", **result, model=MODEL_NAME,
                           label_name=LABELS[result["label"]])


@app.route("/history")
def history():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT ts, filename, label, prob, segs FROM detections ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("history.html", rows=rows)


init_db()


def main():
    app.run(host="0.0.0.0", port=5000, debug=True)

if __name__ == "__main__":
    main()
