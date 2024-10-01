#!/usr/bin/python
"""Main project file, contains everything to generate stereograms."""

import argparse
import json
import os
import re
import codecs
import time
import sys
from random import choice, random

# This "PIL" refers to Pillow, the PIL fork. Check https://pillow.readthedocs.io/en/
import PIL
from PIL import Image as im
from PIL import ImageDraw as imd
from PIL import ImageFilter as imflt
from PIL import ImageFont as imf
from typing import Tuple

from log import Log as log

# Program info
PROGRAM_VERSION = "2.0"

SUPPORTED_IMAGE_EXTENSIONS = [".png", ".jpeg", ".bmp", ".eps", ".gif", ".jpg", ".im", ".msp", ".pcx", ".ppm", ".spider", ".tiff", ".webp", ".xbm"]
DEFAULT_DEPTHTEXT_FONT = "SourceHanSans-Bold.otf"
# 查找所有系统字体的文件路径
FONT_ROOT = "freefont/"

DEFAULT_OUTPUT_EXTENSION = SUPPORTED_IMAGE_EXTENSIONS[0]
# CONSTANTS
DMFOLDER = "depth_maps"
PATTERNFOLDER = "patterns"
SAVEFOLDER = "saved"

# SETTINGS
MAX_DIMENSION = 1500  # px
PATTERN_FRACTION = 8.0
OVERSAMPLE = 1.8
SHIFT_RATIO = 0.3
LEFT_TO_RIGHT = False  # Defines how the pixels will be shifted (left to right or center to sides)
DOT_OVER_PATTERN_PROBABILITY = 0.3  # Defines how often dots are chosen over pattern on random pattern selection

TEXT_FACTOR = 0.8


def show_img(i):
    i.show()


def _hex_color_to_tuple(s):
    """
    Parses a hex color string to a RGB triplet.

    Parameters
    ----------
    s: str
        Valid hex color string. Three-chars supported

    Returns
    -------
    tuple: Equivalent RGB triplet

    """
    if not re.search(r'^#?(?:[0-9a-fA-F]{3}){1,2}$', s):
        return 0, 0, 0
    if len(s) == 3:
        s = "".join(["{}{}".format(c, c) for c in s])
    return tuple(int(c) for c in codecs.decode(s, 'hex'))  # s.decode('hex'))


def make_background(size: Tuple[int, int], filename: str = "", dots_prob: float = None, bg_color: str = "000", dot_colors_string: str = None):
    """
    构建背景图案

    参数
    ----------
    size: tuple(int, int) 深度图的尺寸
    filename: str 图案图片的名称，如果有。如果是点图案则为空
    dots_prob: float 点出现的概率。只有当filename未设置（或等于'dots'）时才有意义
    bg_color: str 颜色的十六进制代码
    dot_colors_string: str 点的颜色字符串，逗号分隔的十六进制颜色代码
    Returns
    -------
    PIL.Image.Image, bool 图像对象及其是否为图片标志
    """
    log.d(f"颜色字符串: {dot_colors_string}")
    pattern_width = int(size[0] / PATTERN_FRACTION)
    # 图案比原始图片稍微长一点，这样在3D中所有东西都能合适（交叉视法会水平压缩图片！）
    i = im.new("RGB", (size[0] + pattern_width, size[1]), color=_hex_color_to_tuple(bg_color))
    i_pix = i.load()
    if filename == "R" and random() < DOT_OVER_PATTERN_PROBABILITY:
        filename = "dots"
    # 从图片加载
    is_image = False
    if filename != "" and filename != "dots":
        pattern = load_file(get_random("pattern") if filename == "R" else filename)
        if pattern is None:
            log.w(f"加载图案 '{filename}' 出错。将使用随机点生成")
            filename = ""
        else:
            is_image = True
            pattern = pattern.resize((pattern_width, int(pattern_width * 1.0 / pattern.size[0]) * pattern.size[1]),
                                     im.LANCZOS)
            # 垂直重复
            region = pattern.crop((0, 0, pattern.size[0], pattern.size[1]))
            y = 0
            while y < i.size[1]:
                i.paste(region, (0, y, pattern.size[0], y + pattern.size[1]))
                y += pattern.size[1]
    # 随机填充
    if filename == "" or filename == "dots":
        for f in range(i.size[1]):
            for c in range(pattern_width):
                if random() < dots_prob:
                    if dot_colors_string is None:
                        i_pix[c, f] = choice([(255, 0, 0), (255, 255, 0), (200, 0, 255)])
                    else:
                        colors = [_hex_color_to_tuple(s) for s in dot_colors_string.split(",")]
                        i_pix[c, f] = choice(colors)

    return i, is_image


def get_random(whatfile: str = "depthmap") -> str:
    """
    从深度图文件夹或图案文件夹中返回一个随机文件

    参数
    ----------
    whatfile: str 指定哪个文件夹

    返回
    -------
    str 随机选择的绝对文件路径
    """
    folder = DMFOLDER if whatfile == "depthmap" else PATTERNFOLDER
    return folder + "/" + choice(os.listdir(folder))


def redistribute_grays(img_object: PIL.Image.Image, gray_height: float) -> PIL.Image.Image:
    """
    对于灰度深度图，压缩灰度范围，使其在0和最大灰度高度之间

    参数
    ----------
    img_object: PIL.Image.Image 打开的图像
    gray_height: float 最大灰度。0=黑色，1=白色

    返回
    -------
    PIL.Image.Image 修改后的图像对象
    """
    if img_object.mode != "L":
        img_object = img_object.convert("L")
    # 确定最小和最大的灰度值
    min_gray = {"point": (0, 0), "value": img_object.getpixel((0, 0))}
    max_gray = {"point": (0, 0), "value": img_object.getpixel((0, 0))}

    for x in range(img_object.size[0]):
        for y in range(img_object.size[1]):
            this_gray = img_object.getpixel((x, y))
            if this_gray > max_gray["value"]:
                max_gray["point"] = (x, y)
                max_gray["value"] = this_gray
            if this_gray < min_gray["value"]:
                min_gray["point"] = (x, y)
                min_gray["value"] = this_gray

    # 转换到新的尺度
    old_min = min_gray["value"]
    old_max = max_gray["value"]
    old_interval = old_max - old_min
    new_min = 0
    new_max = int(255.0 * gray_height)
    new_interval = new_max - new_min
    if old_interval > 0:
        conv_factor = float(new_interval) / float(old_interval)
    else:
        conv_factor = 1.0
    pixels = img_object.load()
    for x in range(img_object.size[0]):
        for y in range(img_object.size[1]):
            pixels[x, y] = int(pixels[x, y] * conv_factor) + new_min
    return img_object


def make_stereogram(parsed_args):
    # 加载或创建立体图深度图
    if parsed_args.text:
        dm_img = make_depth_text(parsed_args.text, parsed_args.font)
    else:
        dm_img = load_file(parsed_args.depthmap, "L")
    # 如果需要的话，应用高斯模糊
    if parsed_args.blur and parsed_args.blur != 0:
        dm_img = dm_img.filter(imflt.GaussianBlur(parsed_args.blur))

    # 重新分配灰度范围（强制深度）
    if parsed_args.text:
        dm_img = redistribute_grays(dm_img, parsed_args.forcedepth if parsed_args.forcedepth is not None else 0.5)
    elif parsed_args.forcedepth:
        dm_img = redistribute_grays(dm_img, parsed_args.forcedepth)

    # 创建空白画布
    pattern_width = (int)(dm_img.size[0] / PATTERN_FRACTION)
    canvas_img = im.new(mode="RGB",
                        size=(dm_img.size[0] + pattern_width, dm_img.size[1]),
                        color=(0, 0, 0) if parsed_args.dot_bg_color is None
                        else _hex_color_to_tuple(parsed_args.dot_bg_color))
    # 创建图案
    pattern_strip_img = im.new(mode="RGB",
                               size=(pattern_width, dm_img.size[1]),
                               color=(0, 0, 0) if parsed_args.dot_bg_color is None
                               else _hex_color_to_tuple(parsed_args.dot_bg_color))
    if parsed_args.pattern:
        # 从文件创建图案
        pattern_raw_img = load_file(parsed_args.pattern)
        p_w = pattern_raw_img.size[0]
        p_h = pattern_raw_img.size[1]
        # 调整到条纹宽度
        pattern_raw_img = pattern_raw_img.resize((pattern_width, (int)((pattern_width * 1.0 / p_w) * p_h)), im.LANCZOS)
        # 垂直重复
        region = pattern_raw_img.crop((0, 0, pattern_raw_img.size[0], pattern_raw_img.size[1]))
        y = 0
        while y < pattern_strip_img.size[1]:
            pattern_strip_img.paste(region, (0, y, pattern_raw_img.size[0], y + pattern_raw_img.size[1]))
            y += pattern_raw_img.size[1]

        # 过采样。更平滑的结果。
        dm_img = dm_img.resize(((int)(dm_img.size[0] * OVERSAMPLE), (int)(dm_img.size[1] * OVERSAMPLE)))
        canvas_img = canvas_img.resize(((int)(canvas_img.size[0] * OVERSAMPLE), (int)(canvas_img.size[1] * OVERSAMPLE)))
        pattern_strip_img = pattern_strip_img.resize(((int)(pattern_strip_img.size[0] * OVERSAMPLE), (int)(pattern_strip_img.size[1] * OVERSAMPLE)))
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
            color_tuples = [_hex_color_to_tuple(hex) for hex in hex_tuples]
        else:
            color_tuples = [(255, 0, 0), (255, 255, 0), (200, 0, 255)]
        log.d("用于点的颜色: {}".format(color_tuples))
        for y in range(pattern_strip_img.size[1]):
            for x in range(pattern_width):
                if random() < dot_prob:
                    pixels[x, y] = choice(color_tuples)

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
        canvas_img = canvas_img.resize(((int)(canvas_img.size[0] / OVERSAMPLE), (int)(canvas_img.size[1] / OVERSAMPLE)),
                                       im.LANCZOS)  # NEAREST, BILINEAR, BICUBIC, LANCZOS
    return canvas_img


def load_font(font_name, font_size):
    try:
        font_path = os.path.join(os.path.dirname(__file__), 'freefont', font_name)
        fnt = imf.truetype(font_path, font_size)
    except IOError:
        print(f"Failed to load font '{font}'. Using default font.")
        fnt = imf.load_default()
    return fnt


def make_depth_text(text, font, canvas_size=(800, 600)):
    """
    创建文字深度图
    """
    # 创建图像（灰度）
    img = PIL.Image.new('L', canvas_size, "black")
    draw = imd.Draw(img)

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


def save_to_file(img_object, output_dir=None):
    """
    尝试保存文件。

    参数
    ----------
    img_object : PIL.Image.Image 要保存的图像对象。
    output_dir : str 保存文件的目录。
    返回
    -------
    tuple(bool, str) 状态：成功则返回True。 附加数据：保存成功时返回图像路径，否则返回失败原因。
    """
    file_ext = DEFAULT_OUTPUT_EXTENSION
    # 尝试按照指定格式保存文件
    if output_dir is None:
        savefolder = SAVEFOLDER
    else:
        savefolder = output_dir
    # 如果目录不存在，则尝试创建目录
    if not os.path.exists(savefolder):
        try:
            os.mkdir(savefolder)
        except IOError as e:
            log.e("无法创建目录: {}".format(e))
            return False, "无法创建输出目录 '{}': {}".format(savefolder, e)
    # 使用日期时间作为文件名的一部分
    outfile_name = u"{date}{ext}".format(
        date=time.strftime("%Y%m%d-%H%M%S", time.localtime()),
        ext=file_ext
    )
    out_path = os.path.join(savefolder, outfile_name)
    try:
        # 尝试保存图像
        r = img_object.save(out_path)
        log.d("文件已保存至 {}".format(out_path))
        return True, out_path
    except IOError as e:
        log.e("保存图像时出错: {}".format(e))
        return False, "无法创建文件 '{}': {}".format(out_path, e)


def load_file(name, type=''):
    """
    加载文件。
    参数
    ----------
    name : str 文件名。
    type : str 图像类型（如 'L' 表示灰度图）。
    返回
    -------
    PIL.Image.Image 或 None  成功加载则返回图像对象，否则返回None。
    """
    try:
        # 尝试打开图像文件
        i = im.open(name)
        if type != "":
            # 如果指定了类型，则转换图像类型
            i = i.convert(type)
    except IOError as msg:
        # 处理打开文件时发生的错误
        log.e("无法加载图像 '{}': {}".format(name, msg))
        return None
    # 如果图像尺寸过大，则调整大小
    if max(i.size) > MAX_DIMENSION:
        max_dim = 0 if i.size[0] > i.size[1] else 1
        old_max = i.size[max_dim]
        new_max = MAX_DIMENSION
        factor = new_max / float(old_max)
        log.d("图像尺寸过大: {}. 按比例调整大小 {}".format(i.size, factor))
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

    arg_parser.add_argument("--blur", "-b", help="高斯模糊程度,图片生成默认2，文本默认4", type=_restricted_blur)
    arg_parser.add_argument("--forcedepth", help="强制使用的最大深度", type=_restricted_unit)
    arg_parser.add_argument("--output", "-o", help="存储结果的目录", type=_existent_directory)
    arg_parser.add_argument("--font", "-f",help="要使用的字体文件。则字体根目录为freefont '{}'".format(FONT_ROOT))

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


# HTTP 响应码类
class _HTTPCode:
    OK = 200
    BAD_REQUEST = 400
    INTERNAL_SERVER_ERROR = 500


# 返回 HTTP 响应
def return_http_response(code, text):
    print(json.dumps({
        "code": code,
        "text": text
    }))


def main(args_list=None):
    # 记录开始生成的信息
    log.i("--- 开始生成 ---")
    # 获取命令行参数
    parsed_args = obtain_args(args_list)
    # 输出参数信息
    log.d("参数: ")
    for key in vars(parsed_args):
        log.d("\t {}: {}".format(key, getattr(parsed_args, key)))
    # 开始计时
    t0 = time.time()
    # 生成立体图
    i = make_stereogram(parsed_args)
    # 如果没有指定输出文件，则展示临时预览
    if not parsed_args.output:
        log.i("过程在 {0:.2f}s 内成功完成".format(time.time() - t0))
        log.i("未指定输出文件。正在显示临时预览")
        show_img(i)
        return
    # 保存到文件
    success, additional_info = save_to_file(i, parsed_args.output)
    # 记录完成情况
    log.d("完成。成功状态: {}, 额外信息: {}".format(success, additional_info))
    # 如果保存失败，则记录错误并返回错误响应
    if not success:
        log.e("过程带有错误完成: '{}'".format(additional_info))
        return_http_response(_HTTPCode.INTERNAL_SERVER_ERROR, additional_info)
    # 否则记录成功信息并返回成功响应
    else:
        log.i("过程在 {0:.2f}s 内成功完成".format(time.time() - t0))
        return_http_response(_HTTPCode.OK, os.path.basename(additional_info))


# 如果是直接运行此脚本，则执行 main 函数
if __name__ == "__main__":
    args_list = ["--text", "钱相皓\n臭宝宝", "--pattern", "patterns/jellybeans2.jpg", "--cross"]
    # args_list = ["--depthmap", "depthmaps/shark.png", "--pattern", "patterns/jellybeans2.jpg", "--cross"]
    main(args_list)

"""
Problems:
When image sharply changes from one depth to a different one, a part of the surface edge repeats to the right and left.
Internet's explanation is that there are some points one eye shouldn't be able to see, but we nonetheless consider them
in the stereogram. They say it can be fixed... but how?
This is called Hidden Surface Removal.
"""

# TODO: Provide option to match pattern height
# TODO: Put generation options as image metadata
