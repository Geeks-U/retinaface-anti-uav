import os
import re
import copy
import json
from typing import Any, TypedDict

import cv2
import numpy as np
from torch.utils.data import Dataset

from src.utils.data_preprocess import generate_triplet_sample


def load_json(json_path: str) -> Any:
    """读取并解析json文件，返回Python对象"""
    if not os.path.exists(json_path):
        print(f"文件不存在: {json_path}")
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"解析JSON文件失败: {e}")
        return None


# 定义配置类型，data_dir改成str
class DatasetConfig(TypedDict):
    data_dir: str
    input_image_size: list[int]
    frame_gap: int
    augment: bool

# 默认配置，data_dir用字符串路径
cfg_dataset_default: DatasetConfig = {
    'data_dir': r'D:\Data\deeplearning\datasets\Anti-UAV\test',
    'input_image_size': [320, 320],
    'frame_gap': 2,
    'augment': False
}

class CustomDataset(Dataset):
    def __init__(self, cfg_dataset: dict | None = None):
        self.cfg: DatasetConfig = copy.deepcopy(cfg_dataset_default)
        if cfg_dataset is not None:
            invalid_keys = set(cfg_dataset.keys()) - set(self.cfg.keys())
            if invalid_keys:
                raise ValueError(f"Invalid config keys: {invalid_keys}")
            self.cfg.update(cfg_dataset)
        self.samples = self.load_annotations()

    def load_annotations(self):
        sequences = []

        for subDir in os.listdir(self.cfg['data_dir']):
            sequence_frames = []
            img_dir = os.path.join(self.cfg['data_dir'], subDir, 'infrared')
            label_path = os.path.join(self.cfg['data_dir'], subDir, 'infrared.json')

            if not os.path.exists(img_dir) or not os.path.exists(label_path):
                continue  # 跳过无效子目录

            annotations = load_json(label_path)
            img_names = sorted(os.listdir(img_dir))

            for img_name in img_names:
                match = re.search(r'I(\d+)\.jpg', img_name)
                if not match:
                    continue

                index = int(match.group(1))
                img_path = os.path.join(img_dir, img_name)

                # 获取对应帧的标注 无目标填充 xlylwh -> xlylxryr
                score = annotations['exist'][index]
                bbox = annotations['gt_rect'][index] if score != 0 else [0, 0, 0, 0]
                bbox[2] += bbox[0]
                bbox[3] += bbox[1]

                # 保存样本信息
                sequence_frames.append([img_path, [[*bbox, score]]])
            sequences.append(sequence_frames)

        # 时序数据构造
        triplet_samples = generate_triplet_sample(sequences, frame_gap=self.cfg['frame_gap'])
        return triplet_samples

    def __len__(self):
        return len(self.samples)

    def to_gray(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray.astype(np.float32)

    def resize_img(self, img, size=None):
        if size is None:
            size = tuple(self.cfg['input_image_size'])
        return cv2.resize(img, size)

    def enhance_brightness(self, gray_img):
        if not self.cfg.get('augment', False):
            return gray_img
        factor = np.random.uniform(0.7, 1.3)
        enhanced = gray_img * factor
        enhanced = np.clip(enhanced, 0, 255)
        return enhanced.astype(np.float32)

    def __getitem__(self, idx):
        img_paths, label = self.samples[idx]  # label格式如 [[x1, y1, x2, y2, score]]
        H_ori, W_ori = cv2.imread(img_paths[0]).shape[:2]  # 原始尺寸

        frames = []
        for p in img_paths:
            gray = self.to_gray(p)
            gray = self.resize_img(gray)
            gray = self.enhance_brightness(gray)
            frames.append(gray)

        diff1 = frames[1] - frames[0]
        diff2 = frames[2] - frames[1]
        gray_t = frames[2]

        # 增加channel维度，变成 (C=1, D=3, H, W)，符合conv3d输入格式
        clip = np.stack([diff1, diff2, gray_t], axis=0).astype(np.float32)
        clip = np.expand_dims(clip, axis=0).astype(np.float32)

        # 标签处理
        bbox = np.array(label, dtype=np.float32)  # (N,5)
        mask = (bbox[:, 2] - bbox[:, 0] >= 1) & (bbox[:, 3] - bbox[:, 1] >= 1)
        bbox = bbox[mask]
        if bbox.shape[0] == 0:
            bbox = np.array([[0, 0, 0, 0, 0]], dtype=np.float32)
        bbox[:, [0, 2]] /= float(W_ori)
        bbox[:, [1, 3]] /= float(H_ori)

        return clip, bbox


# 自定义批次处理函数(默认会将批次label对齐, 该项目每个图象label数量可能不同导致对不齐)
def detection_collate(batch):
    images  = []
    targets = []
    for img, box in batch:
        if len(box)==0:
            continue
        images.append(img)
        targets.append(box)
    images = np.array(images)
    return images, targets


if __name__ == '__main__':
    dataset = CustomDataset({'augment': True})
    c, b = dataset[0]
    print(b)
