import os
import csv
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# 1. Dataset: patch-level motion features (dense sampling)
# -----------------------------

class SpookyPatchTemporalCSV(Dataset):
    """
    Dataset that reads a CSV file with lines: video_path,label

    For each video:
    - load ALL frames (dense), resize to size x size
    - if #frames >= max_frames → use first max_frames frames
    - if #frames <  max_frames → pad last frame
    - compute patch mean differences for each pair of consecutive frames
    """

    def __init__(
        self,
        csv_path: str,
        max_frames: int = 32,
        size: int = 48,
        patch_size: int = 8,
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

        # patch size must divide size
        if size % patch_size != 0:
            raise ValueError(f"size={size} not divisible by patch_size={patch_size}")
        self.patch_size = patch_size
        self.num_patches_h = size // patch_size
        self.num_patches_w = size // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

    def __len__(self):
        return len(self.samples)

    def _load_video_frames(self, path: str) -> List[np.ndarray]:
        """Load ALL video frames (dense sampling)"""
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
        """Keep first max_frames frames; pad if needed."""
        if len(frames) == 0:
            raise RuntimeError("Empty video frames")

        if len(frames) >= self.max_frames:
            used = frames[: self.max_frames]
        else:
            used = frames + [frames[-1]] * (self.max_frames - len(frames))

        return np.stack(used, axis=0)   # [T, H, W]

    def _frames_to_patch_features(self, frames: np.ndarray) -> np.ndarray:
        """
        Convert [T, H, W] frames to patch-level motion features [T-1, 2*num_patches].
        """
        T, H, W = frames.shape
        frames_f = frames.astype(np.float32) / 255.0

        ps = self.patch_size
        ph = self.num_patches_h
        pw = self.num_patches_w

        feats_list = []

        for t in range(1, T):
            prev = frames_f[t - 1]
            curr = frames_f[t]

            prev_means = []
            curr_means = []

            # patch loop
            for ih in range(ph):
                for iw in range(pw):
                    h0 = ih * ps
                    h1 = h0 + ps
                    w0 = iw * ps
                    w1 = w0 + ps

                    prev_patch = prev[h0:h1, w0:w1]
                    curr_patch = curr[h0:h1, w0:w1]

                    prev_means.append(prev_patch.mean())
                    curr_means.append(curr_patch.mean())

            prev_means = np.array(prev_means, np.float32)
            curr_means = np.array(curr_means, np.float32)

            diff = curr_means - prev_means
            abs_diff = np.abs(diff)

            feat = np.concatenate([diff, abs_diff], axis=0)
            feats_list.append(feat)

        return np.stack(feats_list, axis=0)   # [T-1, 2*num_patches]

    def __getitem__(self, idx: int):
        path, label_name = self.samples[idx]

        frames = self._load_video_frames(path)     # load ALL frames
        frames = self._dense_frames(frames)        # dense sampling → [T, H, W]
        feats = self._frames_to_patch_features(frames)  # [T-1, D]

        x = torch.from_numpy(feats)
        y = torch.tensor(self.label_to_idx[label_name], dtype=torch.long)
        return x, y


# -----------------------------
# 2. SNN blocks
# -----------------------------

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = (x > 0).float()
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad = grad_output / ((1.0 + x.abs())**2)
        return grad

spike_fn = SurrogateSpike.apply


class LIFLayer(nn.Module):
    def __init__(self, in_dim, out_dim, tau=10.0, v_threshold=1.0):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.tau = tau
        self.v_threshold = v_threshold

    def forward(self, x):
        T, B, _ = x.shape
        device = x.device
        v = torch.zeros(B, self.fc.out_features, device=device)
        outs = []

        for t in range(T):
            cur = self.fc(x[t])
            v = v + (cur - v) / self.tau
            s = spike_fn(v - self.v_threshold)
            v = v * (1 - s.detach())
            outs.append(s)

        return torch.stack(outs, dim=0)


class SimpleSNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, tau=10.0):
        super().__init__()
        self.lif1 = LIFLayer(in_dim, hidden_dim, tau=tau)
        self.lif2 = LIFLayer(hidden_dim, num_classes, tau=tau)

    def forward(self, x):
        s1 = self.lif1(x)
        s2 = self.lif2(s1)
        logits = s2.mean(dim=0)
        return logits


# -----------------------------
# 3. Training / evaluation
# -----------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    ce = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device).float().transpose(0, 1)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = ce(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_samples += y.size(0)

    return total_loss/total_samples, total_correct/total_samples


def eval_epoch(model, loader, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    ce = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float().transpose(0, 1)
            y = y.to(device)

            logits = model(x)
            loss = ce(logits, y)

            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += y.size(0)

    return total_loss/total_samples, total_correct/total_samples


# -----------------------------
# 4. Main
# -----------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_train", type=str, required=True)
    parser.add_argument("--csv_val", type=str, required=True)
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--patch_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=10.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device)

    train_ds = SpookyPatchTemporalCSV(
        args.csv_train, args.max_frames, args.size, args.patch_size)
    val_ds = SpookyPatchTemporalCSV(
        args.csv_val, args.max_frames, args.size, args.patch_size)

    sample_x, _ = train_ds[0]
    T, D = sample_x.shape
    print(f"T={T}, D={D}, classes={len(train_ds.label_to_idx)}")
    print("feature mean:", sample_x.float().mean().item())
    print("|feature| mean:", sample_x.float().abs().mean().item())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = SimpleSNN(D, args.hidden_dim, len(train_ds.label_to_idx), tau=args.tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, device)

        print(f"Epoch {epoch:02d} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/snn_patch_dense.pt")
    print("Saved model.")


if __name__ == "__main__":
    main()
