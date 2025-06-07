import os
import cv2

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

if __name__ == "__main__":
    convert_all_videos_to_frames(r'D:\Data\deeplearning\datasets\Anti-UAV\test')
