import os
import json
import cv2


def xywh_to_xyxy(box):
    x, y, w, h = box
    x1 = int(x)
    y1 = int(y)
    x2 = int(x + w)
    y2 = int(y + h)
    return [x1, y1, x2, y2]


def generate_fused_images_for_subdir(subdir_path, frame_gap=1):
    json_path = os.path.join(subdir_path, 'infrared.json')
    infrared_dir = os.path.join(subdir_path, 'infrared')
    fused_dir = os.path.join(subdir_path, f'fused_rgb_gap{frame_gap}')

    if not os.path.exists(json_path) or not os.path.exists(infrared_dir):
        print(f"跳过 {subdir_path}, 缺少 json 或 infrared 文件夹")
        return

    os.makedirs(fused_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        anno = json.load(f)

    exist_list = anno['exist']
    gt_rect_list = anno['gt_rect']

    fused_exist = []
    fused_gt_rect = []

    start_frame = 2 * frame_gap
    for i in range(start_frame, len(exist_list)):
        imgs = []
        for j in [i - 2 * frame_gap, i - frame_gap, i]:
            img_name = f'infraredI{j:04d}.jpg'
            img_path = os.path.join(infrared_dir, img_name)
            if not os.path.exists(img_path):
                print(f"警告: 图片不存在 {img_path}")
                break
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"警告: 读取失败 {img_path}")
                break
            imgs.append(img)
        if len(imgs) < 3:
            continue

        fused_img = cv2.merge(imgs)
        fused_name = f'fusedI{i:04d}.jpg'
        fused_path = os.path.join(fused_dir, fused_name)
        cv2.imwrite(fused_path, fused_img)

        fused_exist.append(exist_list[i])
        fused_gt_rect.append(gt_rect_list[i])

    fused_json_path = os.path.join(subdir_path, f'fused_rgb_gap{frame_gap}.json')
    with open(fused_json_path, 'w') as f:
        json.dump({'exist': fused_exist, 'gt_rect': fused_gt_rect}, f)

    print(f"生成 {len(fused_exist)} 张融合图像于 {fused_dir}")


def generate_and_save_fused_images(root_dir, frame_gap=1):
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        generate_fused_images_for_subdir(subdir_path, frame_gap=frame_gap)


if __name__ == "__main__":
    root_dir = r'D:\Data\deeplearning\datasets\Anti-UAV\val'
    generate_and_save_fused_images(root_dir, frame_gap=2)
