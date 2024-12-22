from PIL import Image
import numpy as np


def count_unique_pixel_values(image_path):
    """
    统计 PNG 图片中有几种像素值，以及具体的像素值。

    参数:
    - image_path: str, PNG 图像文件路径。

    返回:
    - unique_values: NumPy 数组，包含所有唯一的像素值。
    """
    # 打开图像并转换为 NumPy 数组
    img = Image.open(image_path)
    pixel_values = np.array(img)

    # 获取所有唯一的像素值
    unique_values = np.unique(pixel_values)

    # 打印结果
    print(f"像素值种类数：{len(unique_values)}")
    print(f"像素值：{unique_values}")

    return unique_values


# 示例用法
image_path = r"C:\Users\jiaqingxiang\Desktop\预测结果\FCN_prediction\prediction_Grazing_000091_4_split0_00000.jpg"  # 替换为你的 PNG 文件路径
unique_values = count_unique_pixel_values(image_path)
