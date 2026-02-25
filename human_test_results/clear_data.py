#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import shutil
import re
import pandas as pd
from tqdm import tqdm
import chardet

# 源数据根目录（每个子目录下有图片和 CSV）
SRC_ROOT = "/home/mingcong/project/embodied/us_proj/data0611/filtered"

# 输出根目录：脚本当前工作目录的 clean/
OUT_ROOT = Path.cwd() / "clean"

# CSV 文件名：优先精确名，其次模糊匹配 *delta*pose*force*.csv（忽略大小写）
CSV_EXACT = "delta_pose_force.csv"
CSV_PATTERN = re.compile(r".*delta.*pose.*force.*\.csv$", re.IGNORECASE)

# 图片扩展名
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 清洗阈值与列
THRESH = 1e-3
# COLS = ["dx","dy","drx","dry","drz"]
COLS = ["drz"]
def detect_encoding(csv_path, sample_size=4096):
    with open(csv_path, "rb") as f:
        raw = f.read(sample_size)
    return chardet.detect(raw)["encoding"] or "utf-8"

def read_csv_auto(csv_path):
    enc = detect_encoding(csv_path)
    # 逗号 -> 制表符 -> 自动推断
    for kwargs in ({}, {"sep": "\t"}, {"sep": None, "engine": "python"}):
        try:
            return pd.read_csv(csv_path, encoding=enc, **kwargs)
        except Exception:
            continue
    return pd.read_csv(csv_path, encoding=enc)

def find_csv_in_dir(d: Path):
    p = d / CSV_EXACT
    if p.is_file():
        return p
    cands = [d / f for f in os.listdir(d) if CSV_PATTERN.match(f)]
    if not cands:
        return None
    cands.sort(key=lambda x: (len(x.name), x.name.lower()))
    return cands[0]

def find_image_path(folder: Path, img_name: str):
    # 1) 直接匹配
    p = folder / img_name
    if p.is_file():
        return p
    # 2) 尝试补扩展名
    base = folder / img_name
    for ext in IMG_EXTS:
        cand = base.with_suffix(ext)
        if cand.is_file():
            return cand
    # 3) 在同目录按 stem 匹配（不递归）
    stem = Path(img_name).stem
    for f in folder.iterdir():
        if f.is_file() and f.suffix.lower() in IMG_EXTS and f.stem == stem:
            return f
    return None

def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    else:
        stem, suf = dst.stem, dst.suffix
        i = 1
        while True:
            alt = dst.with_name(f"{stem}__{i}{suf}")
            if not alt.exists():
                shutil.copy2(src, alt)
                break
            i += 1

def process_one_subdir(subdir: Path, out_root: Path):
    csv_path = find_csv_in_dir(subdir)
    if not csv_path:
        return {"dir": str(subdir), "status": "no_csv"}

    df = read_csv_auto(str(csv_path))
    # 校验必要列
    for c in ["img"] + COLS:
        if c not in df.columns:
            raise ValueError(f"{csv_path} 缺少列: {c}；现有列: {list(df.columns)}")

    # 原逻辑：drop = 全部列的绝对值 < THRESH
    drop_mask = df[COLS].abs().lt(THRESH).all(axis=1)
    # 新逻辑：keep = 非 drop（至少有一列绝对值 >= THRESH）
    keep_mask = ~drop_mask
    df_keep = df.loc[keep_mask].reset_index(drop=True)

    # 若没有保留项，不创建输出目录
    if df_keep.empty:
        return {"dir": str(subdir), "status": "ok", "kept_rows": 0, "copied_imgs": 0, "out_dir": ""}

    # 输出目录：{当前目录}/clean/{子目录}/ 直接放图片与 CSV
    dst_dir = out_root / subdir.name
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = missing = 0
    for _, row in df_keep.iterrows():
        img_name = str(row["img"])
        src_img = find_image_path(subdir, img_name)
        if src_img is None:
            missing += 1
            continue
        dst_img = dst_dir / src_img.name
        copy_file(src_img, dst_img)
        copied += 1

    # 写出 kept CSV 到 {子目录}/ 下
    (dst_dir / "delta_pose_force.csv").write_text(df_keep.to_csv(index=False), encoding="utf-8")

    return {
        "dir": str(subdir),
        "status": "ok",
        "kept_rows": int(len(df_keep)),
        "copied_imgs": int(copied),
        "missing_imgs": int(missing),
        "out_dir": str(dst_dir),
    }

def main():
    src_root = Path(SRC_ROOT)
    if not src_root.is_dir():
        print(f"源目录不存在：{src_root}")
        return

    out_root = OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    subdirs = sorted([p for p in src_root.iterdir() if p.is_dir()])
    if not subdirs:
        print(f"{src_root} 下没有子目录。")
        return

    print(f"将处理 {len(subdirs)} 个子目录。输出到：{out_root}")
    stats = []
    for d in tqdm(subdirs, desc="Export kept (non-trivial) samples"):
        try:
            s = process_one_subdir(d, out_root)
        except Exception as e:
            s = {"dir": str(d), "status": f"error: {e}"}
        stats.append(s)

    ok_stats = [s for s in stats if s.get("status") == "ok"]
    total_kept = sum(s.get("kept_rows", 0) for s in ok_stats)
    total_copied = sum(s.get("copied_imgs", 0) for s in ok_stats)

    print("\n完成：")
    print(f"成功处理: {len(ok_stats)} / {len(stats)} 个子目录")
    print(f"保留行总数: {total_kept}，已复制对应图片: {total_copied}")
    print(f"结果位于：{out_root}/{{子目录}}/")

if __name__ == "__main__":
    main()