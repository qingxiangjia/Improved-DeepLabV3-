import os
import random
from collections import defaultdict

def split_dataset_by_video(image_folder, output_folder, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42):
    """
    根据视频划分图片数据集，将所有视频按照 6:2:2 比例划分为训练集、验证集和测试集，
    并将划分结果分别保存到 train.txt、val.txt 和 test.txt 文件中。

    :param image_folder: 包含所有图片的文件夹路径。
    :param output_folder: 保存划分结果的文件夹路径。
    :param train_ratio: 训练集所占比例，默认为 0.6。
    :param val_ratio: 验证集所占比例，默认为 0.2。
    :param test_ratio: 测试集所占比例，默认为 0.2。
    :param seed: 随机种子的值，确保划分结果可复现。
    """
    # 确保比例和为 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例必须总和为 1.0"

    # 获取文件夹下所有图片文件名
    all_images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    # 如果图片文件夹为空，抛出异常
    if not all_images:
        raise ValueError("图片文件夹为空，请检查路径是否正确，并确保文件夹内包含图片！")

    # 将图片按视频分组
    video_to_images = defaultdict(list)
    for img in all_images:
        video_name = "_".join(img.split("_")[:4])  # 获取视频名称，如 "Drinking_000005_2_split0"
        video_to_images[video_name].append(img)

    # 确保分组的顺序是固定的
    random.seed(seed)
    video_names = sorted(video_to_images.keys())  # 按视频名称排序

    # 计算训练集、验证集和测试集的分割索引
    total_videos = len(video_names)
    train_end = int(total_videos * train_ratio)
    val_end = train_end + int(total_videos * val_ratio)

    # 划分视频为训练集、验证集和测试集
    train_videos = video_names[:train_end]  # 前 60% 的视频用于训练
    val_videos = video_names[train_end:val_end]  # 接下来的 20% 的视频用于验证
    test_videos = video_names[val_end:]  # 最后的 20% 的视频用于测试

    # 根据视频划分图片
    train_images = [img for video in train_videos for img in video_to_images[video]]
    val_images = [img for video in val_videos for img in video_to_images[video]]
    test_images = [img for video in test_videos for img in video_to_images[video]]

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 将图片文件名写入 train.txt
    with open(os.path.join(output_folder, 'train.txt'), 'w') as train_file:
        train_file.writelines(f"{os.path.splitext(img)[0]}\n" for img in train_images)

    # 将图片文件名写入 val.txt
    with open(os.path.join(output_folder, 'val.txt'), 'w') as val_file:
        val_file.writelines(f"{os.path.splitext(img)[0]}\n" for img in val_images)

    # 将图片文件名写入 test.txt
    with open(os.path.join(output_folder, 'test.txt'), 'w') as test_file:
        test_file.writelines(f"{os.path.splitext(img)[0]}\n" for img in test_images)

    # 打印划分结果
    print(f"总视频数: {total_videos}")
    print(f"训练集视频数: {len(train_videos)}，图片数: {len(train_images)}")
    print(f"验证集视频数: {len(val_videos)}，图片数: {len(val_images)}")
    print(f"测试集视频数: {len(test_videos)}，图片数: {len(test_images)}")
    print(f"划分结果已保存到: {output_folder}")


# 使用示例
if __name__ == "__main__":
    # 替换为你的图片文件夹路径
    image_folder = r"E:\CD盘中用空间文件\文件\所有有关yolo+deeplabv3+的改进都在这里\deeplabv3+改进\deeplabv3-plus-pytorch-main\VOCdevkit\cattle13000\JPEGImages"  # 例如 "C:/images"
    # 替换为保存结果的文件夹路径
    output_folder = r"E:\CD盘中用空间文件\文件\所有有关yolo+deeplabv3+的改进都在这里\deeplabv3+改进\deeplabv3-plus-pytorch-main\VOCdevkit\cattle13000\ImageSets\Segmentation"    # 例如 "C:/splits"
    # 调用函数进行划分
    split_dataset_by_video(image_folder, output_folder)
