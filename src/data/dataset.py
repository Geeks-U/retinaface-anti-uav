import os
import re
import copy
import json
import warnings
from typing import Any, TypedDict, Literal

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


# 定义增强选项类型
class EnhanceOpts(TypedDict):
    brightness: bool


# 定义配置类型
class DatasetConfig(TypedDict):
    mode: Literal["train", "test"]
    data_dir: str
    input_image_size: list[int]
    frame_gap: int
    augment: bool
    enhanceOpts: EnhanceOpts

# 默认配置，data_dir用字符串路径
cfg_dataset_default: DatasetConfig = {
    'mode': 'train',
    'data_dir': r'D:\Data\deeplearning\datasets\Anti-UAV\train',
    'input_image_size': [320, 320],
    'frame_gap': 2,
    'augment': True,
    'enhanceOpts': {
        'brightness': True
    }
}

class CustomDataset(Dataset):
    def __init__(self, cfg_dataset: dict | None = None):
        self.cfg: DatasetConfig = copy.deepcopy(cfg_dataset_default)
        if cfg_dataset is not None:
            invalid_keys = set(cfg_dataset.keys()) - set(self.cfg.keys())
            if invalid_keys:
                raise ValueError(f"Invalid config keys: {invalid_keys}")
            self.cfg.update(cfg_dataset)
        self.frames_path = []
        self.samples = self.load_annotations()

    def load_annotations(self):
        res = []

        if self.cfg['mode'] == 'train':
            sequences = []
            for subDir in os.listdir(self.cfg['data_dir']):
                sequence_frames = []
                img_dir = os.path.join(self.cfg['data_dir'], subDir, 'infrared')
                label_path = os.path.join(self.cfg['data_dir'], subDir, 'infrared.json')

                if not os.path.exists(img_dir) or not os.path.exists(label_path):
                    warnings.warn(f"Skipping invalid directory: {img_dir} or label: {label_path}")
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
                    # 质心构造 使用框中心替代
                    x_centroid, y_centroid = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

                    # 保存样本信息
                    sequence_frames.append([img_path, [[*bbox, x_centroid, y_centroid, score]]])
                sequences.append(sequence_frames)
            res = sequences

        elif self.cfg['mode'] == 'test':
            sequence_frames = []
            img_dir = os.path.join(self.cfg['data_dir'], 'infrared')
            if not os.path.exists(img_dir):
                raise FileNotFoundError(f"Image directory not found: {img_dir}")
            img_names = sorted(os.listdir(img_dir))
            for img_name in img_names:
                match = re.search(r'I(\d+)\.jpg', img_name)
                if not match:
                    continue

                index = int(match.group(1))
                img_path = os.path.join(img_dir, img_name)

                # 构造对应帧的标注
                score = 0
                bbox = [0, 0, 0, 0]
                x_centroid, y_centroid = 0, 0

                # 保存样本信息
                sequence_frames.append([img_path, [[*bbox, x_centroid, y_centroid, score]]])
            res = [sequence_frames]

        # 时序数据构造
        triplet_samples = generate_triplet_sample(res, frame_gap=self.cfg['frame_gap'], mode=self.cfg['mode'])
        # for i in triplet_samples[:3]:
        #     print(i)

        return triplet_samples

    def __len__(self):
        return len(self.samples)

    def to_gray(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray.astype(np.float32)

    def resize_img(self, img):
        size = tuple(self.cfg['input_image_size'])
        return cv2.resize(img, size)

    def enhance(self, img, label):
        if not self.cfg['augment'] or self.cfg['mode'] == 'test':
            return img, label
        if self.cfg['enhanceOpts']['brightness']:
            factor = np.random.uniform(0.7, 1.3)
            img = np.clip(img * factor, 0, 255)
        return img, label

    def preprocess_sample(self, img_path: list, label: list):
        # 灰度
        img_gray = list(map(self.to_gray, img_path))
        orig_h, orig_w = img_gray[0].shape

        # resize
        img_resize = list(map(self.resize_img, img_gray))

        # 融合三帧图像为通道维
        fused_img = np.stack([img_resize[0], img_resize[1], img_resize[2]], axis=0)

        # 处理标签 无目标图片(0, 0, 0, 0, 0, 0, 0)作为负样本训练
        bbox_cls = np.array(label, dtype=np.float32)
        mask = (bbox_cls[:, 2] - bbox_cls[:, 0] >= 0) & (bbox_cls[:, 3] - bbox_cls[:, 1] >= 0)
        bbox_cls = bbox_cls[mask]
        bbox_cls[:, [0, 2, 4]] /= orig_w
        bbox_cls[:, [1, 3, 5]] /= orig_h
        bbox_cls[:, :6] = np.clip(bbox_cls[:, :6], 0, 1)

        # 图像增强
        img, label = self.enhance(fused_img, bbox_cls)

        # 归一化和中心化（假设原始像素范围0-255）
        img = img.astype(np.float32) / 255.0 - 0.5

        # 通道差分
        diff_0 = img[1] - img[0]
        diff_1 = img[2] - img[1]
        diff_2 = img[2] - img[0]
        img = np.stack([diff_0, diff_1, diff_2, img[0], img[1], img[2]], axis=0)

        return img, label

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        return self.preprocess_sample(img_path, label)


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
    dataset = CustomDataset({
        'mode': 'train',
        'data_dir': r'D:\Data\deeplearning\datasets\Anti-UAV\test'
    })
    for i in range(1):
        img, l = dataset[i]
        print(img.shape, l)
