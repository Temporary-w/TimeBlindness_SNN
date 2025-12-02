
import os
import csv
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# 1. Dataset: dense frames + temporal-window motion-energy features
# -----------------------------

class SpookyTemporalCSV(Dataset):
    """
    Dataset that reads a CSV file with lines: video_path,label

    For each video:
    - load ALL frames with OpenCV (dense sampling)
    - resize to (size, size)
    - if #frames > max_frames: keep the FIRST max_frames frames (continuous chunk)
    - if #frames < max_frames: pad by repeating the last frame
    - compute low-frequency motion features between consecutive frames:

        blur_prev = GaussianBlur(frame[t-1])
        blur_curr = GaussianBlur(frame[t])
        diff = blur_curr - blur_prev   (signed, capturing direction)
        abs_diff = |diff|              (magnitude)

      得到 per-step 特征序列 [T-1, H*W*2]，然后再做时间窗口平均：
        * 给定 window_size (比如 4)，对连续 window_size 个 diff 做平均
        * 得到 [T - 1 - window_size + 1, H*W*2] 的时间窗特征

      Output x shape: [Tw, H*W*2], continuous-valued.
    """

    def __init__(
        self,
        csv_path: str,
        max_frames: int = 32,
        size: int = 48,
        blur_ksize: int = 9,
        window_size: int = 4,
    ):
        super().__init__()
        self.samples: List[Tuple[str, str]] = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                path, label = row[0], row[1]
                self.samples.append((path, label))

        # Build label mapping
        labels = sorted(list({lab for _, lab in self.samples}))
        self.label_to_idx = {lab: i for i, lab in enumerate(labels)}
        self.idx_to_label = {i: lab for lab, i in self.label_to_idx.items()}

        self.max_frames = max_frames
        self.size = size
        # blur_ksize 必须是奇数
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        self.blur_ksize = blur_ksize

        # 时间窗口长度，至少为 1
        self.window_size = max(1, int(window_size))

    def __len__(self):
        return len(self.samples)

    def _load_video_frames(self, path: str) -> List[np.ndarray]:
        """读取整段视频的所有帧（灰度 + resize）"""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (self.size, self.size))
            frames.append(gray)
        cap.release()
        return frames

    def _dense_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Dense sampling:
        - 如果帧数 >= max_frames：取前 max_frames 帧（连续、不跳帧）
        - 如果帧数 < max_frames：用最后一帧补齐
        返回: [T, H, W]，其中 T == max_frames
        """
        if len(frames) == 0:
            raise RuntimeError("Video has 0 frames after reading.")

        if len(frames) >= self.max_frames:
            used = frames[: self.max_frames]
        else:
            used = frames + [frames[-1]] * (self.max_frames - len(frames))

        arr = np.stack(used, axis=0)  # [T, H, W]
        return arr

    def _frames_to_motion_features(self, frames: np.ndarray) -> np.ndarray:
        """
        Convert [T, H, W] frames to low-frequency motion features
        with temporal windowing: [Tw, H*W*2].

        Steps:
        1) 对所有相邻帧对 (t-1, t) 计算模糊差分:
            diff_list: [T-1, H*W*2]
        2) 时间窗口平均:
            给定 window_size Wsz,
            对每个起点 k 计算:
                mean( diff_list[k : k+Wsz] )
            得到 [T-1 - Wsz + 1, H*W*2]
        """
        T, H, W = frames.shape

        # Normalize to [0,1]
        frames_f = frames.astype(np.float32) / 255.0

        # --- Step 1: per-step motion features ---
        per_step_feats = []

        for t in range(1, T):
            prev = frames_f[t - 1]
            curr = frames_f[t]

            prev_blur = cv2.GaussianBlur(prev, (self.blur_ksize, self.blur_ksize), 0)
            curr_blur = cv2.GaussianBlur(curr, (self.blur_ksize, self.blur_ksize), 0)

            diff = curr_blur - prev_blur          # signed
            abs_diff = np.abs(diff)               # magnitude

            feat_2ch = np.stack([diff, abs_diff], axis=-1)  # [H, W, 2]
            feat_flat = feat_2ch.reshape(H * W * 2)         # [H*W*2]
            per_step_feats.append(feat_flat)

        per_step_feats = np.stack(per_step_feats, axis=0)   # [T-1, H*W*2]

        # 如果 window_size == 1，就不做时间窗口，直接返回
        if self.window_size <= 1:
            return per_step_feats

        # --- Step 2: temporal window averaging ---
        Wsz = self.window_size
        num_steps = per_step_feats.shape[0]

        if num_steps < Wsz:
            # 如果时间步太短，不足以形成一个完整窗口，就直接返回原来的（退化情况）
            return per_step_feats

        window_feats = []
        # k 从 0 到 num_steps - Wsz
        for k in range(num_steps - Wsz + 1):
            window = per_step_feats[k : k + Wsz]       # [Wsz, H*W*2]
            window_mean = window.mean(axis=0)          # [H*W*2]
            window_feats.append(window_mean)

        window_feats = np.stack(window_feats, axis=0)  # [Tw, H*W*2], Tw = num_steps - Wsz + 1
        return window_feats

    def __getitem__(self, idx: int):
        path, label_name = self.samples[idx]
        frames = self._load_video_frames(path)         # all frames
        frames = self._dense_frames(frames)            # [T, H, W] dense + clip/pad
        feats = self._frames_to_motion_features(frames)  # [Tw, N]

        x = torch.from_numpy(feats)  # [Tw, N], continuous
        y = torch.tensor(self.label_to_idx[label_name], dtype=torch.long)
        return x, y


# -----------------------------
# 2. SNN building blocks (with longer tau)
# -----------------------------

class SurrogateSpike(torch.autograd.Function):
    """
    Heaviside step in forward, smooth derivative in backward (surrogate gradient).
    """

    @staticmethod
    def forward(ctx, x):
        out = (x > 0).float()
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # fast sigmoid derivative as surrogate: 1 / (1 + |x|)^2
        grad = grad_output / ((1.0 + x.abs()) ** 2)
        return grad


spike_fn = SurrogateSpike.apply


class LIFLayer(nn.Module):
    """
    Fully-connected LIF layer over time.

    Input:  x of shape [T, B, in_dim]
    Output: spikes of shape [T, B, out_dim]

    tau 控制时间整合的尺度，tau 越大，记忆越长。
    """

    def __init__(self, in_dim: int, out_dim: int, tau: float = 10.0, v_threshold: float = 1.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.tau = tau
        self.v_threshold = v_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, _ = x.shape
        device = x.device
        v = torch.zeros(B, self.fc.out_features, device=device)
        outputs = []

        for t in range(T):
            cur = self.fc(x[t])  # [B, out_dim]
            # leaky integrate with longer tau
            v = v + (cur - v) / self.tau
            s = spike_fn(v - self.v_threshold)
            v = v * (1.0 - s.detach())
            outputs.append(s)

        return torch.stack(outputs, dim=0)  # [T, B, out_dim]


class SimpleSNN(nn.Module):
    """
    - LIF hidden layer
    - LIF output layer
    - Time-average spikes as logits
    输入已经是“时间窗口后的运动特征”，SNN 负责进一步在时间上整合。
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int, tau: float = 10.0):
        super().__init__()
        self.lif1 = LIFLayer(in_dim, hidden_dim, tau=tau)
        self.lif2 = LIFLayer(hidden_dim, num_classes, tau=tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [T, B, in_dim] continuous input
        return: [B, num_classes] logits
        """
        s1 = self.lif1(x)
        s2 = self.lif2(s1)
        logits = s2.mean(dim=0)
        return logits


# -----------------------------
# 3. Training / evaluation loop
# -----------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    ce = nn.CrossEntropyLoss()

    for x, y in loader:
        # x: [B, T, N] → [T, B, N]
        x = x.to(device).float()
        y = y.to(device)
        x = x.transpose(0, 1)

        optimizer.zero_grad()
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += y.size(0)

    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc


def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device)
            x = x.transpose(0, 1)
            logits = model(x)
            loss = ce(logits, y)

            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc


# -----------------------------
# 4. Main
# -----------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Temporal-window SNN demo on SpookyBench-like videos (dense sampling).")
    parser.add_argument("--csv_train", type=str, required=True, help="CSV file listing training videos and labels")
    parser.add_argument("--csv_val", type=str, required=True, help="CSV file listing validation videos and labels")
    parser.add_argument("--max_frames", type=int, default=32, help="Max frames per video (dense, then clip/pad)")
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=10.0, help="LIF time constant (larger = longer memory)")
    parser.add_argument("--blur_ksize", type=int, default=9, help="Gaussian blur kernel size (odd number)")
    parser.add_argument("--window_size", type=int, default=4, help="Temporal window size for motion averaging")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds = SpookyTemporalCSV(
        args.csv_train,
        max_frames=args.max_frames,
        size=args.size,
        blur_ksize=args.blur_ksize,
        window_size=args.window_size,
    )
    val_ds = SpookyTemporalCSV(
        args.csv_val,
        max_frames=args.max_frames,
        size=args.size,
        blur_ksize=args.blur_ksize,
        window_size=args.window_size,
    )

    # Inspect one sample to infer input dim & feature stats
    sample_x, _ = train_ds[0]  # [T, N]
    T, N = sample_x.shape
    num_classes = len(train_ds.label_to_idx)
    print(f"T (timesteps after window): {T}, N (input dim): {N}, num_classes: {num_classes}")
    print(f"Motion feature mean: {sample_x.float().mean().item():.6f}")
    print(f"Motion |feature| mean: {sample_x.float().abs().mean().item():.6f}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = SimpleSNN(in_dim=N, hidden_dim=args.hidden_dim, num_classes=num_classes, tau=args.tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)
        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f}, acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f}, acc {val_acc:.3f}"
        )

    # save model
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = os.path.join("checkpoints", "snn_spooky_temporal_window.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "label_to_idx": train_ds.label_to_idx,
            "idx_to_label": train_ds.idx_to_label,
            "T": T,
            "N": N,
        },
        ckpt_path,
    )
    print("Saved model to", ckpt_path)


if __name__ == "__main__":
    main()
