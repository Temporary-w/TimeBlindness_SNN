import os
import csv

def generate_csv(root_dirs, output_csv):
    """
    root_dirs: 一个列表，包含若干类别文件夹路径，如:
        ["data/circle", "data/square"]
    output_csv: 输出 CSV 的文件名，如 "val.csv"
    """

    rows = []

    for folder in root_dirs:
        # 获取类别名称（文件夹名）
        class_name = os.path.basename(folder.rstrip("/"))

        # 获取文件列表（按名称排序）
        files = sorted([
            f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
        ])

        if len(files) < 2:
            print(f"⚠️ 文件夹 {folder} 中的文件不足 2 个，跳过该类别")
            continue

        # 取最后两个文件 ⭐⭐⭐
        last_two_files = files[-2:]   # <-- 修改的关键行

        for fname in last_two_files:
            full_path = os.path.join(folder, fname)
            rows.append([full_path, class_name])

    # 写入 CSV（无表头）
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"✅ CSV 已生成：{output_csv}   共 {len(rows)} 条数据")


if __name__ == "__main__":
    # 在这里编辑需要扫描的文件夹
    folder_list = [
        "data/spooky_shapes/arrow",
        "data/spooky_shapes/heart"
    ]

    generate_csv(folder_list, "val.csv")
