import os
import argparse
import base64
import time
import re
import codecs
import numpy as np

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from typing import Tuple
from io import BytesIO

# from log import Log as log
from loguru import logger as log

# 系统支持的图片类型
SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpeg", ".bmp", ".eps", ".gif", ".jpg", ".im", ".msp", ".pcx", ".ppm", ".spider", ".tiff", ".webp", ".xbm"]
# 默认的字体名称
DEFAULT_DEPTHTEXT_FONT = "SourceHanSans-Bold.otf"
# 查找所有系统字体的文件路径
FONT_ROOT = "freefont/"
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
    # 加载或创建立体图深度图
    if parsed_args.text:
        dm_img = make_depth_text(parsed_args.text, parsed_args.font, parsed_args.txt_canvas_size)
    else:
        dm_img = load_file(parsed_args.depthmap, "L")
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
        pattern_raw_img = load_file(parsed_args.pattern)
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
    shift_pixels(dm_center_x, dm_img, canvas_img, 1)
    shift_pixels(dm_center_x + pattern_width, dm_img, canvas_img, -1)

    # 回退过采样
    if parsed_args.pattern:
        canvas_img = canvas_img.resize((int(canvas_img.size[0] / OVERSAMPLE), int(canvas_img.size[1] / OVERSAMPLE)),
                                       Image.Resampling.LANCZOS)  # NEAREST, BILINEAR, BICUBIC, LANCZOS
    return canvas_img


def load_font(font_name, font_size):
    try:
        font_path = os.path.join(os.path.dirname(__file__), 'freefont', font_name)
        fnt = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Failed to load font '{font_name}'. Using default font.")
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


def obtain_args(args_list=None):
    """
    获取并解析命令行参数到一个字典中。
    """

    def _restricted_unit(x):
        # 定义一个单位值验证函数，确保值在 0.0 和 1.0 之间
        x = float(x)
        min = 0.0
        max = 1.0
        if x < min or x > max:
            raise argparse.ArgumentTypeError("{} 不在有效范围内 [{}] 至 [{}]".format(x, min, max))
        return x

    def _restricted_blur(x):
        # 定义一个模糊值验证函数，确保值在 0 和 100 之间
        x = int(x)
        min = 0
        max = 100
        if x < min or x > max:
            raise argparse.ArgumentTypeError("{} 不在有效范围内 [{}] 至 [{}]".format(x, min, max))
        return x

    def _restricted_size(x) -> Tuple[int, int]:
        # 一个画布的大小，必须是一个Tuple
        x = x.replace('(', '').replace(')', '').replace('（', '').replace('）', '').replace('，', ',')

        # 使用 split 方法分割字符串，并对每个元素使用 strip 去除空白字符
        parts = [item.strip() for item in x.split(',')]

        # 确保分割后的列表长度为2
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"{x} 必须包含且仅包含两个整数值")

        # 将字符串转换为整数
        try:
            width = int(parts[0] or '0')
            height = int(parts[1] or '0')
        except ValueError:
            raise argparse.ArgumentTypeError(f"{x} 包含非整数值，不是一个有效的元组")

        # 返回一个包含两个整数的元组
        return (width, height)

    def _supported_image_file(filename):
        # 验证给定的文件是否存在且扩展名是否被支持
        if not os.path.exists(filename):
            raise argparse.ArgumentTypeError("文件不存在")
        _, ext = os.path.splitext(filename)
        if filename != "dots" and ext.strip().lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            raise argparse.ArgumentTypeError("文件扩展名不受支持。有效选项为: {}".format(SUPPORTED_IMAGE_EXTENSIONS))
        return filename

    def _existent_directory(dirname):
        # 验证给定的目录是否存在
        if not os.path.isdir(dirname):
            raise argparse.ArgumentTypeError("'{}' 不是一个目录".format(dirname))
        return dirname

    def _valid_color_string(s):
        # 验证颜色字符串是否为有效的十六进制颜色
        if not re.search(r'^#?(?:[0-9a-fA-F]{3}){1,2}$', s):
            raise argparse.ArgumentTypeError("'{}' 不是有效的十六进制颜色".format(s))
        return s

    def _valid_colors_list(s):
        # 验证颜色列表是否为有效的十六进制颜色组合
        colors = s.strip().split(",")
        validated_colors = [_valid_color_string(re.sub(r'x\d+', '', color_string)) for color_string in colors]
        return s

    # 初始化命令行参数解析器
    arg_parser = argparse.ArgumentParser(description="Stereogramaxo: 一个自动立体图生成器")
    # 解析命令行参数或传入的参数列表
    # 添加互斥参数组
    depthmap_arg_group = arg_parser.add_mutually_exclusive_group(required=True)
    depthmap_arg_group.add_argument("--depthmap", "-d", help="深度图图像文件路径", type=_supported_image_file)
    depthmap_arg_group.add_argument("--text", "-t", help="生成带文本的深度图", type=str)

    pattern_arg_group = arg_parser.add_mutually_exclusive_group(required=True)
    pattern_arg_group.add_argument("--dots", help="为背景生成点图案", action="store_true")
    pattern_arg_group.add_argument("--pattern", "-p", help="用于背景图案的图像文件路径", type=_supported_image_file)

    viewmode_arg_group = arg_parser.add_mutually_exclusive_group(required=True)
    viewmode_arg_group.add_argument("--wall", "-w", help="墙眼模式", action="store_true")
    viewmode_arg_group.add_argument("--cross", "-c", help="交叉眼模式", action="store_true")

    dotprops_arg_group = arg_parser.add_argument_group()
    dotprops_arg_group.add_argument("--dot-prob", help="点出现概率", type=_restricted_unit)
    dotprops_arg_group.add_argument("--dot-bg-color", help="背景颜色", type=_valid_color_string)
    dotprops_arg_group.add_argument("--dot-colors",
                                    help="逗号分隔的十六进制颜色列表。支持多重，例如：fff,ff0000x3 表示 ff0000 的数量是 fff 的三倍。",
                                    type=_valid_colors_list)

    arg_parser.add_argument("--txt_canvas_size", "-s", help="生成文本时，默认的画布大小", type=_restricted_size, default=(800, 600))
    arg_parser.add_argument("--blur", "-b", help="高斯模糊程度,图片生成默认2，文本默认4", type=_restricted_blur)
    arg_parser.add_argument("--forcedepth", help="强制使用的最大深度", type=_restricted_unit)
    arg_parser.add_argument("--font", "-f", help="要使用的字体文件。则字体根目录为freefont '{}'".format(FONT_ROOT))

    # 解析命令行参数
    if args_list is None:
        args = arg_parser.parse_args()
    else:
        args = arg_parser.parse_args(args_list)
    # 验证某些参数组合的有效性
    if args.dot_prob and not args.dots:
        arg_parser.error("--dot-prob 只有在设置了 --dots 时才有意义")
    if args.dot_bg_color and not args.dots:
        arg_parser.error("--dot-bg-color 只有在设置了 --dots 时才有意义")
    if args.dot_colors and not args.dots:
        arg_parser.error("--dot-colors 只有在设置了 --dots 时才有意义")
    if not args.blur:
        args.blur = 2 if args.depthmap else 6
    if args.font and not args.text:
        arg_parser.error("--font 只有在使用 --text 时才有意义")

    # 返回解析后的参数
    return args


def stereo_run(args_list=None):
    # 记录开始生成的信息
    log.debug("--- 开始生成 ---")
    # 获取命令行参数
    parsed_args = obtain_args(args_list)
    # 输出参数信息
    log.debug("参数: ")
    for key in vars(parsed_args):
        log.debug("\t {}: {}".format(key, getattr(parsed_args, key)))
    # 开始计时
    t0 = time.time()
    # 生成立体图
    i = make_stereogram(parsed_args)
    # 如果没有指定输出文件，则展示临时预览
    log.info("过程在 {0:.2f}s 内成功完成".format(time.time() - t0))
    log.info("未指定输出文件。正在显示临时预览")
    # show_img(i)
    return success(f"data:image/png;base64,{image_to_base64(i)}")


def success(message: str):
    return {"success": True, "status": 0, "message": message}


def error(message: str):
    return {"success": False, "status": 1, "message": message}


def stereo_run(args_list=None):
    # 记录开始生成的信息
    log.info("--- 开始生成 ---")
    # 获取命令行参数
    parsed_args = obtain_args(args_list)
    # 输出参数信息
    log.debug("参数: ")
    for key in vars(parsed_args):
        log.debug("\t {}: {}".format(key, getattr(parsed_args, key)))
    # 开始计时
    t0 = time.time()
    # 生成立体图
    i = make_stereogram(parsed_args)
    # 如果没有指定输出文件，则展示临时预览
    log.info("过程在 {0:.2f}s 内成功完成".format(time.time() - t0))
    log.info("未指定输出文件。正在显示临时预览")
    i.show()
    return success(f"data:image/png;base64,{image_to_base64(i)}")


# 如果是直接运行此脚本，则执行 main 函数
if __name__ == "__main__":
    args_list = ["--text", "钱相皓\n臭宝宝", "--pattern", "patterns/jellybeans.jpg", "--cross", "--txt_canvas_size", "(800,600)"]
    # args_list = ["--depthmap", "depthmaps/shark.png", "--pattern", "patterns/jellybeans2.jpg", "--cross"]
    log.info(stereo_run(args_list))
