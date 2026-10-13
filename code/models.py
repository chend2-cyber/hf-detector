# -*- coding: utf-8 -*-
"""models.py — 四个对比模型：LSTM / BiLSTM / CNN-LSTM / Transformer
统一约定：输入 (B, 1, 1280)，输出 logits (B, 1)
"""
import torch
import torch.nn as nn

from config import DEVICE, set_seed


class LSTMModel(nn.Module):
    """基准时序模型：单向 LSTM 取最后时间步"""

    def __init__(self, hidden=64, layers=2, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=drop, bidirectional=False)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        x = x.transpose(1, 2)           # (B, 1, 1280) -> (B, 1280, 1)
        out, _ = self.lstm(x)           # (B, 1280, hidden)
        return self.fc(out[:, -1, :])   # 取最后时间步


class BiLSTMModel(nn.Module):
    """双向版：正反两个方向都能看到完整上下文"""

    def __init__(self, hidden=64, layers=2, drop=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=drop, bidirectional=True)
        self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class CNNLSTMModel(nn.Module):
    """先 CNN 抽局部波形形态（P/QRS/T 波），再 LSTM 建模时序依赖"""

    def __init__(self, hidden=64, layers=2, drop=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=drop, bidirectional=False)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.cnn(x)                 # (B, 32, 320)
        x = x.transpose(1, 2)           # (B, 320, 32)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class TransformerModel(nn.Module):
    """注意力模型：先 4x 平均池化降 token 数（注意力是 O(L^2)，1280 点直接算显存吃不消）"""

    def __init__(self, d_model=32, nhead=4, layers=2, drop=0.1):
        super().__init__()
        self.down = nn.AvgPool1d(kernel_size=4)      # 1280 -> 320
        self.embed = nn.Linear(1, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=64, dropout=drop, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.down(x)                 # (B, 1, 320)
        x = x.transpose(1, 2)            # (B, 320, 1)
        x = self.embed(x)                # (B, 320, d_model)
        x = self.encoder(x)
        return self.fc(x.mean(dim=1))    # 时间维平均池化


def build_model(name: str) -> nn.Module:
    models = {
        "lstm": LSTMModel,
        "bilstm": BiLSTMModel,
        "cnn_lstm": CNNLSTMModel,
        "transformer": TransformerModel,
    }
    if name not in models:
        raise ValueError(f"未知模型: {name}，可选 {list(models)}")
    return models[name]().to(DEVICE)


def main():
    """随机 batch 冒烟测试：四个模型输出都应是 (4, 1)"""
    set_seed()
    x = torch.randn(4, 1, 1280).to(DEVICE)
    for name in ("lstm", "bilstm", "cnn_lstm", "transformer"):
        model = build_model(name)
        out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:12s} out={tuple(out.shape)}  params={n_params:,}")

if __name__ == "__main__":
    main()
