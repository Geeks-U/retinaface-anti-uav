import os
import copy
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.datamodule import DataModule
from src.nets.retinaface import Retinaface
from src.utils.anchor import CustomAnchors, calc_raw_bbox, calc_raw_centroid, non_max_suppression


cfg_test_default = {
    'model_path': r'D:\Code\DL\Pytorch\demo\weights\model_20250607_073806_best.pth',
    'input_image_size': [960, 960],
    'cuda': True,
    'variance': [0.1, 0.2],
    'confidence': 0.5
}


class Tester:
    def __init__(self, cfg_tester=None):
        self.cfg = copy.deepcopy(cfg_test_default)
        if cfg_tester is not None:
            self.cfg.update(cfg_tester)

        self.device = 'cuda' if self.cfg['cuda'] and torch.cuda.is_available() else 'cpu'
        if self.cfg['cuda'] and self.device == 'cpu':
            print('CUDA is not available.')

        self.anchors = CustomAnchors(
            cfg_anchor={'input_image_size': self.cfg['input_image_size'],
                        'num_anchor_per_pixel': self.cfg['num_anchor_per_pixel'],
                        'anchor_ratios_per_level': self.cfg['anchor_ratios_per_level']}
        ).get_center_anchors().to(self.device)

        self.model = Retinaface(cfg_model={
            'num_anchor': self.cfg['num_anchor_per_pixel']
        })
        self.model.load_state_dict(torch.load(self.cfg['model_path']))
        self.model.to(self.device)
        self.model.eval()

    def detect_single_video(self, data_dir, orig_img_size=[512, 640], output_json_path=None):
        h, w = orig_img_size
        # 使用 Lightning DataModule 来加载数据
        self.datamodule = DataModule(cfg_datamodule={'batch_size': 4})
        self.datamodule.setup(stage='test',
                              cfg_test={
                                  'input_image_size': self.cfg['input_image_size'],
                                  'data_dir': data_dir
                                  })
        self.test_loader = self.datamodule.test_dataloader()
        frames_pre = []
        for i, (images, targets) in enumerate(self.test_loader):
            images = torch.from_numpy(images).float().to(self.device)

            # outputs['bbox'].shape = torch.Size([B, 37800, 4])
            # outputs['centroid'].shape = torch.Size([B, 37800, 2])
            # outputs['cls'].shape = torch.Size([B, 37800, 2])
            outputs = self.model(images)

            bbox = calc_raw_bbox(outputs['bbox'], self.anchors, self.cfg['variance'])
            scale = torch.tensor([w, h, w, h], dtype=bbox.dtype, device=bbox.device)
            bbox = bbox * scale

            centroid = calc_raw_centroid(outputs['centroid'], self.anchors, self.cfg['variance'])
            scale = torch.tensor([w, h], dtype=bbox.dtype, device=bbox.device)
            centroid = centroid * scale

            cls = F.softmax(outputs['cls'], dim=-1)[:, :, 1:2]

            bbox_centroid_cls = torch.cat([bbox, centroid, cls], dim=-1)
            nms_bbox_centroid_cls = non_max_suppression(bbox_centroid_cls, self.cfg['confidence'])
            for b_c in nms_bbox_centroid_cls:
                frame_pre = []
                if b_c.shape[0] == 0:
                    frame_pre.append([0, 0, 0, 0, 0, 0, 0])
                else:
                    for i in range(b_c.shape[0]):
                        frame_pre.append(b_c[i].tolist())
                frames_pre.append(frame_pre)

        # 获取图片路径 (与dataset加载逻辑保持一致以对齐输入和预测)
        img_dir = os.path.join(data_dir, 'infrared')
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        img_names = sorted(os.listdir(img_dir))
        imgs_path = [os.path.join(img_dir, img_name) for img_name in img_names]

        # 确保数量一致 此处可选择将预测结果持久化
        print(len(imgs_path), len(frames_pre))

        # 可视化生成视频
        save_video_from_frames(imgs_path, frames_pre, output_path='output.mp4', orig_img_size=orig_img_size)

        return


def save_video_from_frames(img_paths, bbox_list, output_path='output.mp4', orig_img_size=(512, 640), fps=10):
    """
    将图片和对应的预测框（包含质心）可视化并合成为视频
    :param img_paths: 图片路径列表
    :param bbox_list: 每帧的预测框列表（每个元素为 [x1, y1, x2, y2, cx, cy, score]）
    :param output_path: 输出视频路径
    :param orig_img_size: 原始图像尺寸 (height, width)
    :param fps: 视频帧率
    """
    if len(img_paths) != len(bbox_list):
        raise ValueError("图像数量与预测框数量不一致")

    h, w = orig_img_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for img_path, bboxes in zip(img_paths, bbox_list):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (w, h))

        for box in bboxes:
            x1, y1, x2, y2, cx, cy, score = box
            if score < 0.1:  # 过滤低置信度框
                continue

            # 绘制框
            color = (0, 255, 0)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # 显示置信度
            cv2.putText(img, f"{score:.2f}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 绘制质心
            cv2.circle(img, (int(cx), int(cy)), 3, (0, 0, 255), -1)

        video_writer.write(img)

    video_writer.release()
    print(f"视频已保存到: {output_path}")


if __name__ == '__main__':
    test = Tester()
    test.detect_single_image(
        image_path=r'D:\Code\DL\Pytorch\retinaface\src\images\29_Students_Schoolkids_Students_Schoolkids_29_60.jpg'
    )
