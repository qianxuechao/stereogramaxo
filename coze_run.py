import random
import os
import re
import base64
import time
import requests
import math
import numpy as np

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from typing import Type, get_args, Any, Tuple, Union, TypeVar, get_args
from io import BytesIO
from dataclasses import dataclass, fields
from loguru import logger as log

# 系统支持的图片类型
SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpeg", ".bmp", ".eps", ".gif", ".jpg", ".im", ".msp", ".pcx", ".ppm", ".spider", ".tiff", ".webp", ".xbm"]
# 模板名称
PATTERN_NAMES = ['arrows.jpg', 'bubbles.jpg', 'comb.gif', 'cool.jpg', 'damask.png', 'gems.jpg', 'grass1.png', 'grass2.png', 'grass.jpg', 'jellybeans2.png',
                 'jellybeans.jpg', 'lava2.jpg', 'lava.jpg', 'leaves2.jpg', 'leaves.jpg', 'neon.jpg', 'paint.jpg', 'stars.jpg', 'stones.jpg', 'straw.jpg',
                 'stripes.jpg', 'tiger.png', 'trippy1.png', 'trippy2.png', 'trippy3.png', 'trippy4.png', 'trippy5.png', 'trippy6.png', 'trippy7.png',
                 'trippy8.png', 'trippy9.png', 'trippy10.png', 'trippy11.png', 'trippy12.png', 'trippy13.png', 'trippy14.png', 'trippy15.png', 'trippy16.png',
                 'trippy17.png', 'trippy18.png', 'trippy19.png', 'trippy20.png', 'trippy21.png', 'trippy22.png', 'trippy23.png', 'trippy24.png', 'trippy25.png',
                 'trippy.png']
# 默认的字体名称
DEFAULT_DEPTHTEXT_FONT = "SourceHanSans-Bold.otf"
# 查找所有系统字体的文件路径
FONT_ROOT = "freefont/"
PATTERN_ROOT = "patterns/"
DEPTHMAP_ROOT = "depthmaps/"
# 文本占据整个画布的比例
TEXT_FACTOR = 0.8

# 最大尺寸，pattern的最大尺寸
MAX_DIMENSION = 1500  # px
# pattern占据的比例
PATTERN_FRACTION = 8.0
# 统一的过采样数值
OVERSAMPLE = 1.8
# 移动的比例
SHIFT_RATIO = 0.3


def image_to_base64(image):
    # 创建一个BytesIO对象
    buffer = BytesIO()

    # 将图像保存到BytesIO对象中
    image.save(buffer, format='PNG')  # 或者其他支持的格式如JPEG等

    # 获取字节数据
    byte_data = buffer.getvalue()

    # 对字节数据进行Base64编码
    base64_data = base64.b64encode(byte_data)

    # 将Base64编码的数据转换成字符串
    base64_string = base64_data.decode('utf-8')

    return base64_string


def hex_color_to_tuple(hex_str):
    """
    将十六进制颜色字符串转换为RGB三元组。

    参数:
    hex_str (str): 十六进制颜色字符串，例如 '#FFAABB'。

    返回:
    tuple: 包含红、绿、蓝三个颜色分量的元组，例如 (255, 170, 187)。
    """
    # 去掉字符串前面的'#'
    hex_str = hex_str.lstrip('#')

    # 将十六进制颜色字符串转换为整数，并切分成RGB三个分量
    return tuple(int(hex_str[i:i + 2], 16) for i in range(0, len(hex_str), 2))


def redistribute_grays(img_object: Image.Image, gray_height: float) -> Image.Image:
    """
    对于灰度深度图，压缩灰度范围，使其在0和最大灰度高度之间。

    参数:
    ----------
    img_object: Image.Image 打开的图像。
    gray_height: float 最大灰度。0=黑色，1=白色。

    返回:
    -------
    Image.Image 修改后的图像对象。
    """
    if img_object.mode != "L":
        img_object = img_object.convert("L")

    # 将图像转换为NumPy数组
    img_array = np.array(img_object)

    # 确定最小和最大的灰度值
    min_gray = np.min(img_array)
    max_gray = np.max(img_array)

    # 计算转换因子
    old_interval = max_gray - min_gray
    new_min = 0
    new_max = int(255.0 * gray_height)
    new_interval = new_max - new_min
    if old_interval > 0:
        conv_factor = float(new_interval) / float(old_interval)
    else:
        conv_factor = 1.0

    # 应用转换因子
    img_array = (img_array - min_gray) * conv_factor + new_min

    # 将结果限制在0到255之间
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    # 将NumPy数组转换回PIL图像
    img_object = Image.fromarray(img_array)

    return img_object


def make_stereogram(parsed_args):
    def download_image(url):
        response = requests.get(url)
        if response.status_code == 200:
            # 使用 BytesIO 处理二进制内容
            image = Image.open(BytesIO(response.content))
            # 计算新的尺寸，确保最长边不超过1920像素
            max_size = 1024
            if image.width > max_size or image.height > max_size:
                ratio = min(max_size / image.width, max_size / image.height)
                new_width = int(image.width * ratio)
                new_height = int(image.height * ratio)
                image.thumbnail((new_width, new_height), resample=Image.LANCZOS)
            # 将图像转换为灰度图
            image = image.convert('L')
            return image
        else:
            raise ValueError(f"下载图片出错,URL:{url},Code: {response.status_code}")

    # 加载或创建立体图深度图
    if parsed_args.text:
        dm_img = make_depth_text(parsed_args.text, parsed_args.font, parsed_args.txt_canvas_size)
    else:
        if parsed_args.depthmap.startswith("http"):
            dm_img = download_image(parsed_args.depthmap)
        else:
            dm_img = load_file(proj_path(DEPTHMAP_ROOT, parsed_args.depthmap), "L")
    # 确认 dm_img 是 PIL.Image.Image 类型
    assert isinstance(dm_img, Image.Image), f"dm_img must be of type PIL.Image.Image, got {type(dm_img)} instead."

    # 如果需要的话，应用高斯模糊
    if parsed_args.blur and parsed_args.blur != 0:
        dm_img = dm_img.filter(ImageFilter.GaussianBlur(parsed_args.blur))

    # 重新分配灰度范围（强制深度）
    if parsed_args.text:
        dm_img = redistribute_grays(dm_img, parsed_args.forcedepth if parsed_args.forcedepth is not None else 0.5)
    elif parsed_args.forcedepth:
        dm_img = redistribute_grays(dm_img, parsed_args.forcedepth)

    # 创建空白画布
    pattern_width = int(dm_img.size[0] / PATTERN_FRACTION)
    canvas_img = Image.new(mode="RGB",
                           size=(dm_img.size[0] + pattern_width, dm_img.size[1]),
                           color=(0, 0, 0) if parsed_args.dot_bg_color is None
                           else hex_color_to_tuple(parsed_args.dot_bg_color))
    # 创建图案
    pattern_strip_img = Image.new(mode="RGB",
                                  size=(pattern_width, dm_img.size[1]),
                                  color=(0, 0, 0) if parsed_args.dot_bg_color is None
                                  else hex_color_to_tuple(parsed_args.dot_bg_color))
    if parsed_args.pattern:
        # 从文件创建图案
        pattern_raw_img = load_file(proj_path(PATTERN_ROOT, parsed_args.pattern))
        # 确认 dm_img 是 PIL.Image.Image 类型
        assert isinstance(pattern_raw_img, Image.Image), f"dm_img must be of type PIL.Image.Image, got {type(dm_img)} instead."
        p_w = pattern_raw_img.size[0]
        p_h = pattern_raw_img.size[1]
        # 调整到条纹宽度
        pattern_raw_img = pattern_raw_img.resize((pattern_width, (int)((pattern_width * 1.0 / p_w) * p_h)), Image.Resampling.LANCZOS)
        # 垂直重复
        region = pattern_raw_img.crop((0, 0, pattern_raw_img.size[0], pattern_raw_img.size[1]))
        y = 0
        while y < pattern_strip_img.size[1]:
            pattern_strip_img.paste(region, (0, y, pattern_raw_img.size[0], y + pattern_raw_img.size[1]))
            y += pattern_raw_img.size[1]

        # 过采样。更平滑的结果。
        dm_img = dm_img.resize((int(dm_img.size[0] * OVERSAMPLE), int(dm_img.size[1] * OVERSAMPLE)))
        canvas_img = canvas_img.resize((int(canvas_img.size[0] * OVERSAMPLE), int(canvas_img.size[1] * OVERSAMPLE)))
        pattern_strip_img = pattern_strip_img.resize((int(pattern_strip_img.size[0] * OVERSAMPLE), int(pattern_strip_img.size[1] * OVERSAMPLE)))
        pattern_width = pattern_strip_img.size[0]

    else:
        # 创建随机点图案
        pixels = pattern_strip_img.load()
        dot_prob = parsed_args.dot_prob if parsed_args.dot_prob else 0.4
        if parsed_args.dot_colors:
            hex_tuples = list()
            for hex_str in parsed_args.dot_colors.split(','):
                if re.match(r'.+x\d+', hex_str):
                    # 倍数
                    factor = int(re.sub(r'.*x', '', hex_str))
                    hex_tuples.extend([re.sub(r'x\d+', '', hex_str)] * factor)
                else:
                    hex_tuples.append(hex_str)
            color_tuples = [hex_color_to_tuple(hex) for hex in hex_tuples]
        else:
            color_tuples = [(255, 0, 0), (255, 255, 0), (200, 0, 255)]
        log.debug("用于点的颜色: {}".format(color_tuples))
        assert pixels is not None, "value should not be None"
        for y in range(pattern_strip_img.size[1]):
            for x in range(pattern_width):
                if random.random() < dot_prob:
                    pixels[x, y] = random.choice(color_tuples)

    # 关键对象：dm_img, pattern_strip_img, canvas_img
    # 开始立体图生成
    def shift_pixels(dm_start_x, depthmap_image_object, canvas_image_object, direction):
        """移动图像像素。direction==1 向右，-1 向左"""
        depth_factor = pattern_width * SHIFT_RATIO
        cv_pixels = canvas_image_object.load()
        while 0 <= dm_start_x < dm_img.size[0]:
            for dm_y in range(depthmap_image_object.size[1]):
                constrained_end = max(0, min(dm_img.size[0] - 1, dm_start_x + direction * pattern_width))
                for dm_x in range(int(dm_start_x), int(constrained_end), direction):
                    dm_pix = dm_img.getpixel((dm_x, dm_y))
                    assert dm_pix is not None, "value should not be None"
                    # 确保 dm_pix 是整数
                    if not isinstance(dm_pix, int):
                        raise TypeError("Pixel value should be an integer")
                    px_shift = int(dm_pix / 255.0 * depth_factor * (1 if parsed_args.wall else -1)) * direction
                    if direction == 1:
                        cv_pixels[dm_x + pattern_width, dm_y] = canvas_img.getpixel((px_shift + dm_x, dm_y))
                    if direction == -1:
                        cv_pixels[dm_x, dm_y] = canvas_img.getpixel((dm_x + pattern_width + px_shift, dm_y))

            dm_start_x += direction * pattern_strip_img.size[0]

    # 粘贴第一个图案
    dm_center_x = dm_img.size[0] / 2
    canvas_img.paste(pattern_strip_img, (int(dm_center_x), 0, int(dm_center_x + pattern_width), canvas_img.size[1]))
    if not parsed_args.wall:
        canvas_img.paste(pattern_strip_img, (int(dm_center_x - pattern_width), 0, int(dm_center_x), canvas_img.size[1]))
    canvas_img = add_watermark(canvas_img, font_size=26, angle=45, watermark_color=(255, 255, 255, 115))
    shift_pixels(dm_center_x, dm_img, canvas_img, 1)
    shift_pixels(dm_center_x + pattern_width, dm_img, canvas_img, -1)

    # 回退过采样
    if parsed_args.pattern:
        canvas_img = canvas_img.resize((int(canvas_img.size[0] / OVERSAMPLE), int(canvas_img.size[1] / OVERSAMPLE)),
                                       Image.Resampling.LANCZOS)  # NEAREST, BILINEAR, BICUBIC, LANCZOS

    return add_watermark(canvas_img, font_size=50, angle=-45, watermark_color=(0, 0, 0, 20))
    # return canvas_img


def add_watermark(canvas_img, watermark_text="扣子智能体三维立体画自助生成", font_size=30, angle=45, watermark_color=(255, 255, 255, 80)):
    """
    给图像添加水印。

    参数:
    canvas_img: Image对象，需要添加水印的图像。
    watermark_text: str，水印文本，默认为"扣子三维立体画自助生成"。
    font_size: int，字体大小，默认为30。
    angle: int，水印旋转角度，默认为45度。
    watermark_color: tuple，水印颜色，默认为半透明白色。

    返回:
    Image对象，添加水印后的图像。
    """
    line_space = 30
    column_space = 50

    # 加载字体（需要一个字体文件）
    font = load_font(DEFAULT_DEPTHTEXT_FONT, font_size)
    # 创建绘图对象
    draw = ImageDraw.Draw(canvas_img)
    # 获取文本尺寸
    tl, tt, tr, tb = draw.multiline_textbbox((0, 0), watermark_text, font=font, spacing=line_space, align='center')
    text_width = int(tr - tl)
    text_height = int(tb - tt)

    # 确定水印旋转后的最小包围矩形
    temp_img = Image.new('RGBA', (text_width, text_height), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_img)
    temp_draw.multiline_text((0, 0), watermark_text, font=font, fill=watermark_color, spacing=line_space, align='center')
    temp_img = temp_img.rotate(angle, expand=True)
    rotated_box_w, rotated_box_h = temp_img.size
    rotated_box_w = rotated_box_w + column_space
    rotated_box_h = rotated_box_h + line_space

    # 在图片上绘制旋转后的文本
    for x in range(0, canvas_img.width, rotated_box_w):
        for y in range(0, canvas_img.height, rotated_box_h):
            # 确保不会超出边界
            if x + rotated_box_w <= canvas_img.width and y + rotated_box_h <= canvas_img.height:
                # 将临时图像粘贴到原图
                canvas_img.paste(temp_img, (x, y), temp_img)

    return canvas_img


def load_font(font_name, font_size):
    try:
        font_path = proj_path(FONT_ROOT, font_name)
        fnt = ImageFont.truetype(font_path, font_size)
    except IOError:
        log.error(f"Failed to load font '{font_name}'. Using default font.")
        fnt = ImageFont.load_default()
    return fnt


def make_depth_text(text, font, canvas_size=(800, 600)):
    """
    创建文字深度图
    """
    # 创建图像（灰度）
    img = Image.new('L', canvas_size, "black")
    draw = ImageDraw.Draw(img)

    if font is None:
        font = DEFAULT_DEPTHTEXT_FONT

    # 创建参数字典
    text_params = {
        'xy': (0, 0),
        'text': text,
        'font': None,
        'align': 'left',
        'direction': None,
        'features': None,
        'language': None,
        'spacing': 8,
        'anchor': None,
        'stroke_width': 3,
    }

    # 动态调整字体大小
    font_size = 1
    # 尝试使用指定的字体名称创建字体对象
    fnt = load_font(font, font_size)
    text_params['font'] = fnt
    tl, tt, tr, tb = draw.multiline_textbbox(**text_params)

    while (tr - tl) < canvas_size[0] * TEXT_FACTOR and (tb - tt) < canvas_size[1] * TEXT_FACTOR:
        font_size += 1
        fnt = load_font(font, font_size)
        text_params['font'] = fnt
        tl, tt, tr, tb = draw.multiline_textbbox(**text_params)

    # 计算文本的位置以居中
    text_position = (
        canvas_size[0] // 2 - (tr - tl) // 2,
        canvas_size[1] // 2 - (tb - tt) // 2
    )

    text_params['xy'] = text_position
    text_params['fill'] = 255
    # 绘制文本
    draw.multiline_text(**text_params)

    return img


def load_file(name, type=''):
    """
    加载文件。
    参数
    ----------
    name : str 文件名。
    type : str 图像类型（如 'L' 表示灰度图）。
    返回
    -------
    Image.Image 或 None  成功加载则返回图像对象，否则返回None。
    """
    try:
        # 尝试打开图像文件
        i = Image.open(name)
        if type != "":
            # 如果指定了类型，则转换图像类型
            i = i.convert(type)
    except IOError as msg:
        # 处理打开文件时发生的错误
        log.error("无法加载图像 '{}': {}".format(name, msg))
        return None
    # 如果图像尺寸过大，则调整大小
    if max(i.size) > MAX_DIMENSION:
        max_dim = 0 if i.size[0] > i.size[1] else 1
        old_max = i.size[max_dim]
        new_max = MAX_DIMENSION
        factor = new_max / float(old_max)
        log.debug("图像尺寸过大: {}. 按比例调整大小 {}".format(i.size, factor))
        i = i.resize((int(i.size[0] * factor), int(i.size[1] * factor)))
    return i


def success(msg: str):
    return {"success": True, "status": 1, "message": msg}


def error(msg: str):
    return {"success": False, "status": 1, "message": msg}


@dataclass
class StereoParam:
    depthmap: Union[str, None] = None
    text: Union[str, None] = None
    dots: bool = False
    pattern: Union[str, None] = None
    wall: bool = False
    dot_prob: float = 0.0
    dot_bg_color: Union[str, None] = None
    dot_colors: Union[str, None] = None
    txt_canvas_size: Tuple[int, int] = (800, 600)
    blur: int = 2
    forcedepth: Union[float, None] = None
    font: Union[str, None] = None

    # 查询属性类型
    @classmethod
    def get_attribute_type(cls, attr_name):
        # 从类的 __annotations__ 字典中查询属性类型
        return cls.__annotations__.get(attr_name, None)

    # 根据属性类型动态赋值
    def set_attribute(self, attr_name, value):
        attr_type = self.get_attribute_type(attr_name)
        if attr_type is None:
            raise AttributeError(f"Class {StereoParam.__name__} does not have an attribute named '{attr_name}'")

        # 处理 Union 类型
        if hasattr(attr_type, '__origin__') and attr_type.__origin__ is Union:
            union_args = get_args(attr_type)
            if value is None and None in union_args:
                value = None
            else:
                # 尝试使用非 None 类型转换值
                for arg in union_args:
                    if arg is not None:
                        try:
                            value = self.convert_value(arg, value)
                            break
                        except (ValueError, TypeError):
                            continue
                else:
                    raise ValueError(f"Cannot convert value '{value}' to any type in Union[{union_args}]")
        else:
            value = self.convert_value(attr_type, value)
        # 设置属性
        setattr(self, attr_name, value)

    @staticmethod
    def convert_value(attr_type: Type, value: Any) -> Any:
        if attr_type is bool:
            return value.lower() in ['true', '1', 'yes']
        elif attr_type is int:
            return int(value)
        elif attr_type is float:
            return float(value)
        elif attr_type is tuple or attr_type is Tuple:
            # 如果 attr_type 是非泛型的 tuple 或 Tuple
            return tuple(map(int, value.strip("()").split(','))) if '(' in value else attr_type(value)
        elif hasattr(attr_type, '__origin__') and (attr_type.__origin__ is tuple or attr_type.__origin__ is Tuple):
            # 如果 attr_type 是 Tuple 泛型类型
            tuple_args = get_args(attr_type)
            if tuple_args:
                parsed_values = re.findall(r'\d+', value.strip("()"))
                if len(parsed_values) == len(tuple_args):
                    return tuple(map(lambda t, v: t(v), tuple_args, parsed_values))
                else:
                    raise ValueError(f"Incorrect number of values for tuple: expected {len(tuple_args)}, got {len(parsed_values)}")
            else:
                # 如果没有具体类型参数，则作为普通字符串处理
                return tuple(map(int, value.strip("()").split(',')))
        else:
            return attr_type(value)


def ensure_font():
    font_dir = proj_path(FONT_ROOT)
    if not os.path.exists(font_dir):
        os.makedirs(font_dir)
    if not os.path.exists(proj_path(font_dir, DEFAULT_DEPTHTEXT_FONT)):
        # 发送GET请求获取图片内容
        response = requests.get(f"https://cdn.lzpkj.com/coze/{DEFAULT_DEPTHTEXT_FONT}")
        # 检查请求是否成功
        if response.status_code == 200:
            # 打开一个文件以二进制写入模式
            with open(proj_path(font_dir, DEFAULT_DEPTHTEXT_FONT), 'wb') as file:
                # 将图片内容写入文件
                file.write(response.content)
            log.info("字体文件下载成功！")
        else:
            log.error("字体下载失败，状态码:", response.status_code)
    else:
        # 获取文件大小
        file_size = os.path.getsize(proj_path(font_dir, DEFAULT_DEPTHTEXT_FONT))
        log.info(f"字体文件已存在，大小为：{file_size} 字节")


def ensure_pattern():
    pattern_dir = proj_path(PATTERN_ROOT)
    if not os.path.exists(pattern_dir):
        os.makedirs(pattern_dir)
    for pn in PATTERN_NAMES:
        if not os.path.exists(proj_path(pattern_dir, pn)):
            # 发送GET请求获取图片内容
            response = requests.get(f"https://cdn.lzpkj.com/coze/{pn}")
            # 检查请求是否成功
            if response.status_code == 200:
                # 打开一个文件以二进制写入模式
                with open(proj_path(pattern_dir, pn), 'wb') as file:
                    # 将图片内容写入文件
                    file.write(response.content)


def stereo_run(params: StereoParam):
    # 记录开始生成的信息
    log.info("--- 开始生成 ---")
    # 获取命令行参数
    # 输出参数信息
    log.debug("参数: ")
    for key in vars(params):
        log.debug("\t {}: {}".format(key, getattr(params, key)))
    # 开始计时
    t0 = time.time()
    # 生成立体图
    i = make_stereogram(params)
    # 如果没有指定输出文件，则展示临时预览
    log.info("过程在 {0:.2f}s 内成功完成".format(time.time() - t0))
    log.info("未指定输出文件。正在显示临时预览")
    i.show()
    return success(f"data:image/png;base64,{image_to_base64(i)}")


def proj_path(*args):
    # 获取当前工作目录的绝对路径
    current_dir = os.path.abspath(".")

    # 检查第一个参数是否为绝对路径
    if args and os.path.isabs(args[0]):
        # 如果是绝对路径并且与当前工作目录相同，则移除第一个参数
        return os.path.join(*args)

    # 将当前工作目录与传入的路径段组合
    return os.path.join(current_dir, *args)


"""
Each file needs to export a function named `handler`. This function is the entrance to the Tool.

Parameters:
args: parameters of the entry function.
args.input - input parameters, you can get test input value by args.input.xxx.
args.logger - logger instance used to print logs, injected by runtime.

Remember to fill in input/output in Metadata, it helps LLM to recognize and use tool.

Return:
The return data of the function, which should match the declared output parameters.
"""


def check_param(param: StereoParam):
    if param.text:
        param.blur = 6
    else:
        param.blur = 2


# def handler(args: Args[Input]) -> Output:
if __name__ == '__main__':
    # 确保字体文件存在
    ensure_font()

    # 确保图片模板存在
    ensure_pattern()

    # return {"success":True,"status":1,"message":f"图片模板也有{file_names}"}
    # 如果是直接运行此脚本，则执行 main 函数
    params = StereoParam()
    # 示例使用
    attr_name = "depthmap"
    value = "https://p6-bot-sign.byteimg.com/tos-cn-i-v4nquku3lp/0fc0a0f7b5b94a79b742ef14a12611f3.png~tplv-v4nquku3lp-image.image?rk3s=68e6b6b5&x-expires=1730487359&x-signature=GDzfVUzkcrq%2FQkcm9ptTRHOd8lE%3D"
    params.set_attribute(attr_name, value)

    attr_name = "pattern"
    value = "jellybeans.jpg"
    params.set_attribute(attr_name, value)

    attr_name = "wall"
    value = "false"
    params.set_attribute(attr_name, value)

    attr_name = "txt_canvas_size"
    value = "(800,600)"
    params.set_attribute(attr_name, value)

    check_param(params)
    img_base = stereo_run(params)
    # return img_base
    # 如果没有指定输出文件，则展示临时预览
    log.info(img_base)

    # return {"success": True, "status": 1, "message": f"不运行咋样"}
