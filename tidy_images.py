from PIL import Image, ImageDraw, ImageFont
import os
import re
from loguru import logger as log


def resize_image(image, size=(256, 256)):
    """强制转换图片至指定尺寸"""
    return image.resize(size, Image.Resampling.BICUBIC)


def create_image_with_filename(image_path, font_path, font_size=20, size=(256, 256)):
    img = resize_image(Image.open(image_path), size=size)
    width, height = img.size
    # 增加额外的空间用于文本显示，可以根据需要调整这个值
    extra_space_for_text = font_size + 10  # 假设增加了10像素的行间距
    new_img = Image.new('RGB', (width, height + extra_space_for_text), color='white')
    new_img.paste(img, (0, 0))
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(new_img)
    file_name = os.path.basename(image_path)
    tl, tt, tr, tb = draw.textbbox((0, 0), file_name, font=font)
    text_width = tr - tl
    text_height = tb - tt
    x = (width - text_width) // 2
    y = height  # 文本放在图片下方
    draw.text((x, y), file_name, fill='black', font=font)
    return new_img


def grid_combine_images(images, cols=8, rows=6, size=(256, 256), spacing=10):
    # 确保有足够的图像填充网格
    required_images = cols * rows
    if len(images) < required_images:
        raise ValueError(f"Need at least {required_images} images to fill grid.")

    # 获取第一张图像的尺寸
    img_width, img_height = images[0].size

    # 计算总的宽度和高度，包括间距
    combined_img_width = img_width * cols + (cols + 1) * spacing
    combined_img_height = img_height * rows + (rows + 1) * spacing
    combined_img = Image.new('RGB', (combined_img_width, combined_img_height), color='white')

    # 粘贴每个图像到最终图像
    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            if index < len(images):
                img = images[index]
                combined_img.paste(img, (col * (img_width + spacing) + spacing, row * (img_height + spacing) + spacing))

    return combined_img


def natural_keys(text):
    """
    自定义排序键函数，用于自然排序。
    """

    def atoi(text):
        return int(text) if text.isdigit() else text

    return [atoi(c) for c in re.split(r'(\d+)', text)]


# def main(folder_path, output_path, font_path, size=(256, 256), spacing=10):
#     # 获取文件夹中的所有图像文件，并按文件名排序
#     image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
#     image_files.sort(key=natural_keys)  # 使用自然排序
#
#     images_with_filenames = []
#     for image_file in image_files:
#         image_path = os.path.join(folder_path, image_file)
#         images_with_filenames.append(create_image_with_filename(image_path, font_path, size=size))
#
#     final_image = grid_combine_images(images_with_filenames, cols=8, rows=6, size=size, spacing=spacing)
#     final_image.save(output_path)

def main(folder_path, output_path, font_path, size=(256, 256), spacing=10):
    # 获取文件夹中的所有图像文件，并按文件名排序
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    image_files.sort(key=natural_keys)  # 使用自然排序

    file_names = []
    for image_file in image_files:
        file_names.append(image_file)
    log.info(file_names)


if __name__ == '__main__':
    folder_path = './patterns'
    output_path = './patterns/combined_image.png'
    font_path = './freefont/SourceHanSans-Bold.otf'
    size = (256, 256)  # 图片尺寸
    spacing = 10  # 图片之间的间距

    main(folder_path, output_path, font_path, size=size, spacing=spacing)
