import os
import copy
import warnings

import cv2
import numpy as np


def convert_all_videos_to_frames(root_dir):
    """
    遍历 root_dir 下所有目录中的 .mp4 视频文件，将其帧保存为 jpg 图片。
    每个视频的帧保存在与该视频同名的文件夹下，命名格式为 videoNameI0000.jpg。
    """
    # 收集所有 .mp4 文件路径
    video_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.mp4'):
                full_path = os.path.join(dirpath, filename)
                video_paths.append(full_path)

    print(f"共找到 {len(video_paths)} 个视频文件")

    for idx, video_path in enumerate(video_paths):
        print(f"[{idx + 1}/{len(video_paths)}] 处理视频: {video_path}")
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        success, frame = cap.read()

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(os.path.dirname(video_path), video_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建目录: {output_dir}")

        while success:
            frame_filename = f"{video_name}I{str(frame_count).zfill(4)}.jpg"
            save_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(save_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            frame_count += 1
            success, frame = cap.read()

        cap.release()
        print(f"{video_name} 提取完成，共 {frame_count} 帧")

    print("全部视频处理完成。")


def generate_triplet_sample(samples: list, frame_gap: int=2, mode: str='train'):
    # 测试阶段数据填充
    if mode != 'train':
        first_sample = samples[0][0]
        samples[0][:0] = [copy.deepcopy(first_sample) for _ in range(2 * frame_gap)]

    triplet_samples = []
    for v in samples:
        frame_start_index = 2 * frame_gap
        if frame_start_index > len(v):
            warnings.warn(
                f"样本帧数不足：需要至少 {frame_start_index + 1} 帧，但当前只有 {len(v)} 帧。",
                stacklevel=2
            )
            continue
        for i in range(frame_start_index, len(v)):
            pathList = [v[i - 2*frame_gap][0], v[i - frame_gap][0], v[i][0]]
            triplet_samples.append([pathList, copy.deepcopy(v[i][1])])

    return triplet_samples


def is_pseudo_color_image_from_path(path: str, threshold: float = 0.98) -> bool:
    """
    从图像路径判断是否为伪色彩图像（RGB 三通道是否高度相关）

    参数:
        path: str，图像文件路径
        threshold: float，通道之间的相关性阈值，默认 0.98

    返回:
        bool: 是否为伪色彩图像
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")

    img = cv2.imread(path)  # BGR 读取
    if img is None:
        raise ValueError(f"无法读取图像: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("图像不是 RGB 三通道")

    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    rg = np.corrcoef(r.ravel(), g.ravel())[0, 1]
    rb = np.corrcoef(r.ravel(), b.ravel())[0, 1]
    gb = np.corrcoef(g.ravel(), b.ravel())[0, 1]

    is_pseudo = rg > threshold and rb > threshold and gb > threshold
    return is_pseudo


if __name__ == "__main__":
    # convert_all_videos_to_frames(r'D:\Data\deeplearning\datasets\Anti-UAV\test')
    print(is_pseudo_color_image_from_path(r'D:\Data\deeplearning\datasets\Anti-UAV\demo\infrared\infraredI0007.jpg'))
