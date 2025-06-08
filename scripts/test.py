import os
import json
from src.test.tester import Tester


def detect_images_in_dir_to_json(tester, image_dir, output_json_path, image_suffixes=('.jpg', '.png', '.jpeg')):
    results = {}

    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith(image_suffixes)
    ]

    for image_name in sorted(image_files):
        image_path = os.path.join(image_dir, image_name)

        try:
            # 获取检测结果
            _, boxes_with_scores = tester.detect_single_image(image_input=image_path, return_image=True)

            # 转为可序列化格式
            result_list = [
                {
                    "bbox": box,        # [x1, y1, x2, y2]
                    "score": round(score, 4)
                }
                for box, score in boxes_with_scores
            ]

            results[image_name] = result_list
            print(f"✓ 检测完成: {image_name}，检测到 {len(result_list)} 个目标")

        except Exception as e:
            print(f"× 检测失败: {image_name}，错误: {e}")

    # 写入 JSON 文件
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ 所有检测结果已保存至: {output_json_path}")


if __name__ == '__main__':
    # 当前脚本文件的绝对路径
    current_file = os.path.abspath(__file__)
    # 回到项目根目录 （假设脚本在 scripts 目录下）
    base_dir = os.path.dirname(os.path.dirname(current_file))

    model_path = os.path.join(base_dir, 'weights', 'model_20250608_185248_last.pth')
    cfg_tester = {
        'model_path': model_path,
        'input_image_size': [640, 640]
    }

    data_dir = r'D:\Data\deeplearning\datasets\Anti-UAV\val\20190925_152412_1_3'
    test = Tester(cfg_tester=cfg_tester)
    test.detect_single_video(data_dir)

