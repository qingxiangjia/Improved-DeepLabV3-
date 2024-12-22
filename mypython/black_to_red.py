from PIL import Image
import numpy as np
import os

#将黑白图像转为黑红图像
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

    # 将牛（像素值为 1）设置为红色 (255, 0, 0) 绿色(0,255,0)
    rgb_image[mask_array == 0] = [0, 255, 0]  # 纯绿色

    rgb_image[mask_array == 1] = [0, 0, 0]  # 黑色
    # 背景（像素值为 0）保持为黑色（默认）

    # 转换回 PIL 图像
    result_image = Image.fromarray(rgb_image)

    # 保存结果
    result_image.save(output_path)
    print(f"转换完成，结果保存到: {output_path}")



def convert_and_save_binary_image(input_path, output_path):
    """
    将灰度图像中的牛（像素值255）转换为1，背景（像素值0）保持为0，
    并以灰度格式保存图像，确保在查看时0和1都显示为黑色。

    参数：
    - input_path: str，输入的灰度图像文件路径。
    - output_path: str，转换后的输出文件路径。
    """
    # 打开图像，并确保它是灰度模式
    mask = Image.open(input_path).convert('L')

    # 将图像转换为 NumPy 数组
    mask_array = np.array(mask)

    # 将255转换为1
    binary_image = (mask_array // 255).astype(np.uint8)

    # 转换回PIL图像
    result_image = Image.fromarray(binary_image)

    # 保存结果
    result_image.save(output_path, format='PNG')
    print(f"转换完成，结果保存到: {output_path}")

#将RGB转为灰度图像
def convert_image(input_path, output_path):
    # 加载图像
    img = Image.open(input_path)
    img = img.convert('RGB')
    data = np.array(img)

    # 创建一个空的灰度图像数组
    gray_image = np.zeros((data.shape[0], data.shape[1]), dtype=np.uint8)

    # 根据RGB值设置灰度图像的像素值
    # 黑色像素设置为0，非黑色像素设置为255
    black = [0, 0, 0]
    mask_black = np.all(data == black, axis=-1)
    gray_image[~mask_black] = 1  # 非黑色像素设置为255

    # 保存为灰度图像
    gray_img = Image.fromarray(gray_image, 'L')
    gray_img.save(output_path, format='PNG')
    # print(f"转换完成，结果保存到: {output_path}")


# 示例用法
if __name__ == "__main__":
    input_dir = r"C:\Users\jiaqingxiang\Desktop\预测结果\detection-results"
    output_dir = r"C:\Users\jiaqingxiang\Desktop\预测结果\improved_deeplabv3+_predtiction"
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)


            # convert_mask_to_red(input_path, output_path)

            #0和255转为0和1
            # convert_and_save_binary_image(input_path, output_path)
            #
            # convert_image(input_path, output_path)



