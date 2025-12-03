# 📘 SpookyBench SNN Demo

基于 **SpookyBench** 高噪声视频基准测试的 **时序窗口 + SNN（LIF）** 分类模型。
本项目探索：

> **为何人眼能识别 Time Blindness（动态图形错觉），而大模型无法识别？**
> **能否通过 SNN 的时间整合能力模拟人类视觉？**

本仓库包含：数据生成脚本、SNN 训练脚本、patch-level 与 temporal-motion 特征提取方法，以及可复现的训练流程。

---

## 🌟 1. 核心思想（Core Idea）

本项目提供两个关键模型结构：

### **① Temporal Window Motion SNN（主模型）**

* Dense 读取视频所有帧
* 经过 Gaussian Blur（降噪）
* 相邻帧差分：**diff + abs_diff**
* 应用 **时间窗口（Temporal Window）** 平滑运动趋势
* 输入两层 LIF 神经元做时间整合

### **② Patch-Level SNN（局部感受野模型）**

* 将每帧划分为多个 patch
* 对每个 patch 求均值（模拟视觉皮层局部感受野）
* 计算 patch-level 运动变化
* 输入两层 LIF，提取低频稳定结构

这些方法模拟了人类视觉在高噪声环境中的“时间整合 + 空间聚合”能力。

---

## 📂 2. 数据准备（SpookyBench Shapes）

下载 SpookyBench 数据集并解压：

```bash
wget https://huggingface.co/datasets/timeblindness/spooky-bench/resolve/main/spooky_bench.zip
unzip spooky_bench.zip -d data
```

本项目使用其中的 **Shapes** 子集，例如：

```
data/spooky_bench/Shapes/arrow/*.mp4
data/spooky_bench/Shapes/heart/*.mp4
```

你可以挑选部分视频组成训练/验证集。

---

## 📝 3. 生成训练与验证 CSV

本仓库包含两个脚本：

### **train_csv.py**

* 遍历指定类别文件夹
* 自动跳过每类 *最后两个视频*
* 生成训练集 `train.csv`

### **val_csv.py**

* 读取每类的 *最后两个视频*
* 生成验证集 `val.csv`

**CSV 格式：**

```
video_path,label_name
data/spooky_shapes/arrow/video_001.mp4,arrow
data/spooky_shapes/heart/video_003.mp4,heart
```

---

## 🧠 4. 主要训练文件（模型说明）

### **4.1 snn_temporal_window.py（主模型）**

功能：

* Dense 读取视频帧
* Gaussian blur 过滤高频噪声
* 计算帧差 diff / abs_diff
* 使用时间窗口做 motion smoothing
* 输入两层 LIF：

  * LIF(N → hidden_dim)
  * LIF(hidden_dim → num_classes)
* 对所有时间步的脉冲做平均以输出分类结果

适合处理 **只有时间维度才能解码的隐藏图形**。

---

### **4.2 snn_patch_level.py（Patch-Level 模型）**

功能：

* 每帧划分为多个 patch（例如 48×48 → 8×8 patch）
* 对每个 patch 求亮度均值
* 计算 patch 级别的运动变化
* 通过两层 LIF 进行时序整合与分类

适合探索视觉皮层局部分区对稳定结构的提取能力。

---

## ▶️ 5. 运行训练脚本（Training Commands）

确保安装：

```bash
pip install torch opencv-python numpy
```

---

### **5.1 运行 Temporal Window SNN**

```bash
python snn_temporal_window.py \
  --csv_train train.csv \
  --csv_val val.csv \
  --epochs 10 \
  --batch_size 4 \
  --max_frames 32 \
  --window_size 4 \
  --size 48
```

---

### **5.2 运行 Patch-Level SNN**

```bash
python snn_patch_level.py \
  --csv_train train.csv \
  --csv_val val.csv \
  --epochs 10 \
  --batch_size 4 \
  --size 48 \
  --patch_size 8
```

---

## 📊 6. 输出示例

```
Using device: cpu
T (timesteps after window): 29, N: 4608
Motion feature mean: 0.0163
Epoch 01 | train acc 0.545 | val acc 0.500
Epoch 02 | train acc 0.636 | val acc 0.667
...
Saved model to checkpoints/snn_spooky_temporal_window.pt
```

---

## 📄 7. 项目结构

```
SpookySNN/
│
├── data/
│   └── spooky_shapes/
│       ├── arrow/
│       └── heart/
│
├── train_csv.py
├── val_csv.py
│
├── snn_temporal_window.py   # 主训练文件
├── snn_patch_level.py       # patch 版本
│
├── checkpoints/
└── README.md
```

---

## 🎯 8. 总结

本项目实现了一个能够在高噪声视频中复现 **类人视觉识别能力** 的 SNN 系统：
通过 **空间降噪（blur / patch）与时间整合（temporal window / LIF）**，
模型能够从“肉眼才可见”的隐藏动态中提取形状信息。
