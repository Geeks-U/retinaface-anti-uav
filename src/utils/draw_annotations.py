import os
import json
import cv2


def draw_annotations_from_json(demo_dir):
    """
    根据 detect_result.json 中的 bbox 标注，绘制图像并保存
    """
    infrared_dir = os.path.join(demo_dir, 'infrared')
    json_path = os.path.join(demo_dir, 'detect_result.json')
    output_dir = os.path.join(demo_dir, 'annotated')
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        detections = json.load(f)

    for fused_filename, boxes in detections.items():
        # 替换文件名前缀 fused -> infrared
        infrared_filename = fused_filename.replace("fused", "infrared")
        img_path = os.path.join(infrared_dir, infrared_filename)

        if not os.path.exists(img_path):
            print(f"图像 {infrared_filename} 未找到，跳过。")
            continue

        img = cv2.imread(img_path)

        for box in boxes:
            x1, y1, x2, y2 = box['bbox']
            score = box['score']

            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            label = f"{score:.2f}"
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 255, 0), thickness=1)

        output_path = os.path.join(output_dir, fused_filename)
        cv2.imwrite(output_path, img)
        print(f"保存标注图像到: {output_path}")


def images_to_video(image_dir, output_path, fps=10):
    """
    将图像目录合成为视频，按文件名排序
    """
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not image_files:
        print("未找到任何图片")
        return

    first_image_path = os.path.join(image_dir, image_files[0])
    first_frame = cv2.imread(first_image_path)
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"无法读取 {filename}，跳过")
            continue
        video_writer.write(frame)

    video_writer.release()
    print(f"视频已保存到: {output_path}")


if __name__ == "__main__":
    # 设置你的 demo 路径
    demo_dir = r'D:\Data\deeplearning\datasets\Anti-UAV\demo'

    # draw_annotations_from_json(demo_dir)

    annotated_dir = os.path.join(demo_dir, 'annotated')
    video_output_path = os.path.join(demo_dir, 'annotated_video.mp4')

    images_to_video(annotated_dir, video_output_path, fps=10)
