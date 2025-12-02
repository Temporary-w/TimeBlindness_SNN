import os
import csv

def generate_csv(root_dirs, output_csv):
    """
    root_dirs: 一个列表，包含若干个类别文件夹路径，如:
        ["data/circle", "data/square"]
    output_csv: 输出 CSV 路径，如 "train.csv"
    """
    
    rows = []

    for folder in root_dirs:
        # 获取类别名称（上级文件夹名）
        class_name = os.path.basename(folder.rstrip("/"))

        # 列出当前文件夹全部文件
        files = sorted([
            f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
        ])

        if len(files) <= 2:
            print(f"⚠️ 文件夹 {folder} 中的文件不足以跳过最后两个，已跳过该文件夹")
            continue

        # 跳过最后两个文件
        files_to_use = files[:-2]   # ⭐⭐ 改这里：跳过最后 2 个 ⭐⭐

        for fname in files_to_use:
            full_path = os.path.join(folder, fname)
            rows.append([full_path, class_name])

    # 写入 CSV (无表头)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"✅ CSV 已生成：{output_csv}   共 {len(rows)} 条数据")


if __name__ == "__main__":
    # 你可以在这里修改需要扫描的文件夹
    folder_list = [
        "data/spooky_shapes/arrow",
        "data/spooky_shapes/heart"
    ]

    generate_csv(folder_list, "train.csv")
