from pathlib import Path
import copy

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2

cfg_dataset_default = {
    'data_dir': r'D:\Data\deeplearning\datasets\Anti-UAV\val',
    'input_image_size': [320, 320],
    'augment': False
}

class CustomDataset(Dataset):
    def __init__(self, cfg_dataset=None):
        self.cfg = copy.deepcopy(cfg_dataset_default)
        if cfg_dataset is not None:
            self.cfg.update(cfg_dataset)
        self.data_dir = Path(self.cfg['data_dir'])
        self.input_image_size = self.cfg['input_image_size']

        self.samples = build_dataset_info(self.data_dir)

    # 输入: relative_path, [[x_lt_bbox, y_lt_bbox, w_bbox, h_bbox, x0_ldm, y0_ldm, score0_ldm, ..., score], ...]
    # 输出: relative_path, [[x_lt_bbox, y_lt_bbox, x_rb_bbox, y_rb_bbox, x_ldm, y_ldm, ..., score], ...]
    # 有人脸有关键点 score=1
    # 有人脸无关键点 score=-1，ldm值为-1表示无人脸关键点，则将对应坐标置0
    # 无人脸 score=0
    def _load_annotations(self):
        samples = []
        with open(self.label_path, 'r') as f:
            lines = f.readlines()

        current_img = None
        current_labels = []

        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                if current_img is not None:
                    samples.append((current_img, current_labels))
                current_img = line[2:]
                current_labels = []
            elif line:
                parts = list(map(float, line.split()))
                box = parts[:4]
                box = [parts[0], parts[1], parts[0] + parts[2], parts[1] + parts[3]]
                landm = []
                if parts[4] > 0:
                    for i in range(5):
                        landm += parts[4 + (i * 3):6 + (i * 3)]
                    score = [1]
                else:
                    landm = [0 for i in range(10)]
                    score = [-1]

                label = box + landm + score
                current_labels.append(label)

        if current_img is not None:
            samples.append((current_img, current_labels))

        return samples

    def __len__(self):
        return len(self.samples)

    # 输入: 索引
    # 输出: np.array类型的image, label
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path)
        label = np.array(label, dtype=np.float32)

        image, label = self._sample_transform(image=image, labels=label, input_shape=self.input_image_size)

        return image, label

    # 输入: image(PIL打开), label(np.array类型)(corner_px)
    # 输出: np.array类型的image, label(corner_percent)
    def _sample_transform(self, image, labels, input_shape):
        img_w, img_h = image.size
        input_h, input_w = input_shape

        labels = np.array(labels, dtype=np.float32)  # (N,5)

        # 过滤掉无效框：比如坐标非法的框（宽高<=1）
        valid_size_mask = (labels[:, 2] > labels[:, 0] + 1) & (labels[:, 3] > labels[:, 1] + 1)

        labels = labels[valid_size_mask]

        if labels.shape[0] > 0:
            # 归一化坐标（坐标无效的也会被裁剪到0~1范围）
            labels[:, [0, 2]] /= img_w
            labels[:, [1, 3]] /= img_h
            np.clip(labels[:, :4], 0, 1, out=labels[:, :4])
        else:
            labels = np.array([[0, 0, 0, 0, 0]], dtype=np.float32)

        # 图像缩放 + 均值中心化 + 转置
        new_image = np.array(image.resize((input_w, input_h), Image.BICUBIC), dtype=np.float32)
        mean = np.array([127.5, 127.5, 127.5], dtype=np.float32)
        new_image -= mean
        new_image = new_image.transpose(2, 0, 1)

        return new_image, labels


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


import os
import json

def build_dataset_info(root_dir):
    dataset = []

    for sequence in os.listdir(root_dir):
        sequence_path = os.path.join(root_dir, sequence)
        if not os.path.isdir(sequence_path):
            continue

        json_path = os.path.join(sequence_path, 'fused_rgb_gap2.json')
        image_dir = os.path.join(sequence_path, 'fused_rgb_gap2')

        if not os.path.exists(json_path) or not os.path.exists(image_dir):
            continue

        with open(json_path, 'r') as f:
            info = json.load(f)

        exist_flags = info.get('exist', [])
        gt_rects = info.get('gt_rect', [])

        for idx, (exist, rect) in enumerate(zip(exist_flags, gt_rects)):
            image_name = f"fusedI{idx+4:04d}.jpg"  # 注意这里加了 +4
            image_path = os.path.join(image_dir, image_name)
            if len(rect) != 4 or exist==0:
                rect = [0, 0, 0, 0]
                exist = 0
            else:
                rect[2] += rect[0]
                rect[3] += rect[1]
            if not os.path.exists(image_path):
                continue

            dataset.append([image_path, [[*rect, exist]]])

    return dataset


import cv2
import matplotlib.pyplot as plt

def draw_bbox_r_channel(image_path, bbox, color=255, thickness=2):
    """
    只显示图像的红色通道，并在上面画出目标框

    参数:
        image_path (str): 图像路径
        bbox (list or tuple): [x, y, w, h]，左上角 + 宽高
        color (int): 框的灰度值（0-255），默认白色
        thickness (int): 框的线条粗细
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"图像读取失败: {image_path}")
        return

    # 提取 R 通道（OpenCV 图像是 BGR 顺序）
    r_channel = img[:, :, 2].copy()  # R 通道是 index 2

    # 在 R 通道上画框（直接在灰度图上画）
    x, y, xr, yr = map(int, bbox)
    cv2.rectangle(r_channel, (x, y), (xr, yr), color, thickness)

    # 显示
    plt.imshow(r_channel, cmap='gray')
    plt.title(f"R Channel with Box: {bbox}")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    res = CustomDataset()
    print(len(res))
    i, l = res[0]
    print(i.shape)
    draw_bbox_r_channel(res.samples[1000][0], res.samples[1000][1][0][:4])
