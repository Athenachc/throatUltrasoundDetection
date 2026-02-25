import os
from glob import glob
from datetime import datetime
import numpy as np
from scipy.spatial.transform import Rotation as R
import csv
from shutil import copy2

def timestamp_to_sec(ts):
    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
    return dt.timestamp()

def timestamp_to_sec_under(ts):
    # 20250625_174122951
    dt = datetime.strptime(ts, "%Y%m%d_%H%M%S%f")
    return dt.timestamp()

def timestamp_to_sec_second(ts):
    # 20250625_174122951
    dt = datetime.strptime(ts, "%Y-%m-%d_%H_%M_%S.%f")
    return dt.timestamp()

def parse_img_time(fname):
    """
    支持两种图片文件名格式:
    1. 20250625_174122951.jpg
    2. 2025-06-11 13:51:28.595.jpg
    3. 2025-06-11_13_51_28.595.jpg
    """
    try:
        # 尝试新格式
        return timestamp_to_sec_under(fname)
    except Exception:
        # 尝试老格式
        # 补全毫秒为6位，兼容如 .595 这种
        try:
            if len(fname) == 23:  # '2025-06-11 13:51:28.595'
                fname += '000'
            return timestamp_to_sec(fname)
        except Exception:
            # 尝试第三种格式
            try:
                if len(fname) == 23:  # '2025-06-11_13_51_28.595'
                    fname += '000'
                return timestamp_to_sec_second(fname)
            except Exception as e2:
                raise ValueError(f"无法解析时间戳：{fname}")

def parse_pose_file(txt_path):
    with open(txt_path, 'r') as f:
        pose_lines = f.readlines()
    pose_records = []
    for line in pose_lines:
        first_split = line.strip().split(' ', 2)
        ts_str = first_split[0] + ' ' + first_split[1]
        ts_sec = timestamp_to_sec(ts_str)
        nums = []
        for part in line.strip().split(','):
            nums += [float(x) for x in part.strip().split() if x.replace('.', '', 1).replace('e-', '', 1).replace('-', '', 1).isdigit()]
        pos = np.array(nums[0:3])
        rot_mat = np.array(nums[3:12]).reshape(3,3)
        force = np.array(nums[12:15])
        pose_records.append((ts_sec, pos, rot_mat, force))
    return pose_records

def find_nearest_pose_idx(img_time, pose_times):
    idx = np.abs(pose_times - img_time).argmin()
    return idx

def relative_pose(pos1, rot1, pos2, rot2):
    R1 = np.array(rot1)
    R2 = np.array(rot2)
    R_rel = R1.T @ R2
    t_rel = R1.T @ (np.array(pos2) - np.array(pos1))
    rvec_rel = R.from_matrix(R_rel).as_rotvec()
    return np.concatenate([t_rel, rvec_rel])

def rotmat_to_rvec(rot_mat):
    """将旋转矩阵转旋转向量"""
    return R.from_matrix(rot_mat).as_rotvec()

def process_sub_folder(sub_folder, output_root):
    txt_files = glob(os.path.join(sub_folder, '*.txt'))
    if len(txt_files) == 0:
        print(f'No txt file in {sub_folder}, skipped.')
        return
    txt_path = txt_files[0]
    pose_records = parse_pose_file(txt_path)
    pose_times = np.array([r[0] for r in pose_records])

    img_files = sorted(glob(os.path.join(sub_folder, '*.jpg')))
    img_times = []
    for img_path in img_files:
        fname = os.path.basename(img_path).replace('.jpg','')
        try:
            img_times.append(parse_img_time(fname))
        except Exception as e:
            print(f"Image {img_path} name parse error: {e}")
            img_times.append(None)

    out_folder = os.path.join(output_root, os.path.basename(sub_folder))
    os.makedirs(out_folder, exist_ok=True)

    results = []
    kept = 0
    last_kept_img_time = 0.0
    for i in range(len(img_files) - 1):
        img_time = img_times[i]
        nxt_img_time = img_times[i+1]
        if img_time is None or nxt_img_time is None:
            continue

        # 过滤掉时间间隔小于0.1秒的图片
        if (img_time - last_kept_img_time) < 0.1 and last_kept_img_time != 0.0:
            continue

        pose_idx = find_nearest_pose_idx(img_time, pose_times)
        nxt_pose_idx = find_nearest_pose_idx(nxt_img_time, pose_times)

        pos1, rot1, force1 = pose_records[pose_idx][1], pose_records[pose_idx][2], pose_records[pose_idx][3]
        pos2, rot2, force2 = pose_records[nxt_pose_idx][1], pose_records[nxt_pose_idx][2], pose_records[nxt_pose_idx][3]

        if force1[2] > -1.0 or force2[2] > -1.0:
            continue

        delta = relative_pose(pos1, rot1, pos2, rot2)
        euler1 = R.from_matrix(rot1).as_euler('xyz')

        out_img_path = os.path.join(out_folder, os.path.basename(img_files[i]))
        copy2(img_files[i], out_img_path)
        results.append([
            os.path.basename(img_files[i]),
            *pos1.tolist(), *euler1.tolist(),
            *delta.tolist(),
            *force1.tolist()
        ])
        kept += 1

        last_kept_img_time = img_time

    if results:
        out_csv = os.path.join(out_folder, 'delta_pose_force.csv')
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['img', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'fx', 'fy', 'fz'])
            writer.writerows(results)
        print(f"{os.path.basename(sub_folder)}: kept {kept} samples, csv saved at {out_csv}")
        
if __name__ == "__main__":
    print("Processing started...")
    input_root = './data'
    output_root = './filtered'
    os.makedirs(output_root, exist_ok=True)
    all_folders = glob(os.path.join(input_root, '202507151531*'))
    all_sub_folders = [f for f in all_folders if os.path.isdir(f)]
    all_sub_folders.sort()
    print(f'Found {len(all_sub_folders)} day folders.')
    for sub_folder in all_sub_folders:
        process_sub_folder(sub_folder, output_root)