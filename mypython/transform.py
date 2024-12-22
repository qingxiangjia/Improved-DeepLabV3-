from PIL import Image
import numpy as np
import os
import cv2


def extract_frames_from_video(video_path, output_dir):
    """
    将 MP4 视频分解为 PNG 图片帧。
    参数:
    - video_path: str, 输入视频的路径。
    - output_dir: str, 输出 PNG 图片帧的目录。
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 保存帧为 PNG 格式
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1

    cap.release()
    print(f"视频帧提取完成，共提取 {frame_idx} 帧。")

def modify_mask_color(input_dir, output_dir):
    """
    修改 PNG 图片中的牛像素值为红色 (255, 0, 0)。
    参数:
    - input_dir: str, 原始 PNG 图片目录。
    - output_dir: str, 修改后的 PNG 图片目录。
    """
    os.makedirs(output_dir, exist_ok=True)
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".png"):
            img_path = os.path.join(input_dir, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图
            # 创建 RGB 图像
            rgb_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            # 背景为黑色，牛的像素改为红色
            rgb_img[img == 255] = [0, 0, 255]  # 红色为 (BGR: [0, 0, 255])
            # 保存修改后的图像
            output_path = os.path.join(output_dir, file_name)
            cv2.imwrite(output_path, rgb_img)

    print("所有帧的颜色修改完成。")

def reconstruct_video_from_frames(input_dir, output_video_path, fps=30):
    """
    将 PNG 图片帧重构为 MP4 视频。
    参数:
    - input_dir: str, 修改后的 PNG 图片目录。
    - output_video_path: str, 输出视频的路径。
    - fps: int, 输出视频的帧率。
    """
    frame_list = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".png")])
    if not frame_list:
        print("没有找到图片帧，无法生成视频。")
        return

    # 获取第一张图片的尺寸
    first_frame = cv2.imread(frame_list[0])
    height, width, layers = first_frame.shape

    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 MP4 格式
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_path in frame_list:
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()
    print(f"视频重构完成，保存为: {output_video_path}")


def convert_mask_to_red(input_path, output_path):
    """
    将牛的像素值（值为1）转换为红色（[255, 0, 0]），背景保持为黑色。

    参数：
    - input_path: str，输入的二分类 mask 图像文件路径。
    - output_path: str，转换后的输出文件路径。
    """
    # 打开 PNG 图像
    mask = Image.open(input_path).convert('L')  # 将图像转换为灰度模式

    # 将图像转换为 NumPy 数组
    mask_array = np.array(mask)

    # 创建 RGB 空白图像（与输入大小相同）
    rgb_image = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)

    # 将牛（像素值为 1）设置为红色 (255, 0, 0)
    rgb_image[mask_array == 1] = [255, 0, 0]  # 红色
    # 背景（像素值为 0）保持为黑色（默认）

    # 转换回 PIL 图像
    result_image = Image.fromarray(rgb_image)

    # 保存结果
    result_image.save(output_path)
    print(f"转换完成，结果保存到: {output_path}")

def convert_white_to_black(input_path, output_path):
    """
    将牛的像素值（值为255）转换为黑色（值为1），背景保持为黑色。

    参数：
    - input_path: str，输入的二分类 mask 图像文件路径。
    - output_path: str，转换后的输出文件路径。
    """
    # 打开 PNG 图像
    mask = Image.open(input_path).convert('L')  # 将图像转换为灰度模式

    # 将图像转换为 NumPy 数组
    mask_array = np.array(mask)

    # 创建 RGB 空白图像（与输入大小相同）
    rgb_image = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)

    # 将牛（像素值为 1）设置为红色 (255, 0, 0)
    rgb_image[mask_array == 255 ] = 1  # 黑色

    # 背景（像素值为 0）保持为黑色（默认）

    # 转换回 PIL 图像
    result_image = Image.fromarray(rgb_image)

    # 保存结果
    result_image.save(output_path)
    print(f"转换完成，结果保存到: {output_path}")

# 示例用法
if __name__ == "__main__":
    # # 黑黑png转为黑红
    # input_dir = "/Users/wsy/Desktop/数据集转换/json_mask/Video_transform/imgout_masks3"
    # output_dir = "imgout_masks6/"
    # os.makedirs(output_dir, exist_ok=True)
    #
    # for filename in os.listdir(input_dir):
    #     if filename.endswith(".png"):
    #         input_path = os.path.join(input_dir, filename)
    #         output_path = os.path.join(output_dir, filename)
    #         convert_mask_to_red(input_path, output_path)


    # 黑白png转黑红mp4
    temp_frames_dir = r"C:\Users\jiaqingxiang\Desktop\预测结果\1"           # 临时存储提取的帧的目录
    modified_frames_dir = "modified_frames"  # 修改颜色后的帧的目录
    output_video_path = r"C:\Users\jiaqingxiang\Desktop\预测结果\1_transform.mp4"   # 输出视频路径
    # 步骤 2: 修改帧的颜色
    # modify_mask_color(temp_frames_dir, modified_frames_dir)
    # 步骤 3: 将帧重构为视频
    reconstruct_video_from_frames(temp_frames_dir, output_video_path)



    # # 黑白png转为黑黑
    # input_dir = "/Users/wsy/Desktop/数据集转换/json_mask/Video_transform/1"
    # output_dir = "imgout_masks3/"
    # os.makedirs(output_dir, exist_ok=True)
    #
    # for filename in os.listdir(input_dir):
    #     if filename.endswith(".png"):
    #         input_path = os.path.join(input_dir, filename)
    #         output_path = os.path.join(output_dir, filename)
    #         convert_white_to_black(input_path, output_path)
