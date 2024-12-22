import os
import cv2
from collections import defaultdict

# 图片所在的文件夹路径
image_folder = "C:/Users/jiaqingxiang/Desktop/predict_results/dataset_test_groundtruth_mask"
# 输出视频的文件夹路径
output_folder = "C:/Users/jiaqingxiang/Desktop/predict_results/dataset_test_groundtruth_mask_video"
# 视频帧率
frame_rate = 3

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 创建一个字典，将图片按照视频名称分组
video_groups = defaultdict(list)

# 遍历图片文件夹中的所有文件
for filename in sorted(os.listdir(image_folder)):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # 提取视频名称部分（例如 Grazing_000091_4_split0）
        video_name = "_".join(filename.split("_")[:4])
        # 将图片路径添加到对应视频组中
        video_groups[video_name].append(os.path.join(image_folder, filename))

# 合成视频
for video_name, image_paths in video_groups.items():
    # 获取图片的宽高（以第一张图片为准）
    first_image = cv2.imread(image_paths[0])
    height, width, _ = first_image.shape

    # 定义输出视频路径
    output_video_path = os.path.join(output_folder, f"{video_name}.mp4")

    # 创建视频写入对象
    video_writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),  # 视频编码器
        frame_rate,
        (width, height)
    )

    # 将图片写入视频
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # 释放视频写入对象
    video_writer.release()
    print(f"视频已生成: {output_video_path}")

print("所有视频已合成完成！")
