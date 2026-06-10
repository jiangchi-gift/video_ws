import logging
from typing import Tuple, Sequence, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageTransforms:
    """常用图像变换与预处理工具类。

    设计目标：安全校验输入、接受配置参数、在内存或时间受限场景下保持可控。
    """

    @staticmethod
    def _check_img(img: np.ndarray):
        if img is None:
            raise ValueError("输入图像为空")
        if not isinstance(img, np.ndarray):
            raise TypeError("img 必须为 numpy.ndarray")

    @staticmethod
    def to_gray(img: np.ndarray) -> np.ndarray:
        ImageTransforms._check_img(img)
        if img.ndim == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def to_hsv(img: np.ndarray) -> np.ndarray:
        ImageTransforms._check_img(img)
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    @staticmethod
    def hsv_range_mask(img: np.ndarray, lower: Sequence[int] = (0, 0, 0), upper: Sequence[int] = (255, 255, 255),
                       morph: bool = True, ksize: int = 5) -> np.ndarray:
        """返回 HSV 掩码；可选形态学降噪（开/闭运算），在嘈杂环境下更稳健。"""
        ImageTransforms._check_img(img)
        hsv = ImageTransforms.to_hsv(img)
        lower = np.array(lower, dtype=np.int32)
        upper = np.array(upper, dtype=np.int32)
        mask = cv2.inRange(hsv, lower, upper)
        if morph:
            kernel = np.ones((ksize, ksize), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    @staticmethod
    def resize(img: np.ndarray, size: Tuple[int, int] = (320, 240), interpolation: int = cv2.INTER_AREA) -> np.ndarray:
        ImageTransforms._check_img(img)
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError("size 必须为正整数元组")
        return cv2.resize(img, size, interpolation=interpolation)

    @staticmethod
    def resize_scale(img: np.ndarray, fx: float = 1.0, fy: float = 1.0, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        ImageTransforms._check_img(img)
        if fx <= 0 or fy <= 0:
            raise ValueError("缩放因子必须为正数")
        return cv2.resize(img, None, fx=fx, fy=fy, interpolation=interpolation)

    @staticmethod
    def rotate(img: np.ndarray, angle: float = 0.0, center: Optional[Tuple[int, int]] = None, scale: float = 1.0) -> np.ndarray:
        ImageTransforms._check_img(img)
        h, w = img.shape[:2]
        if center is None:
            center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR)

    @staticmethod
    def flip_horizontal(img: np.ndarray) -> np.ndarray:
        ImageTransforms._check_img(img)
        return cv2.flip(img, 1)

    @staticmethod
    def flip_vertical(img: np.ndarray) -> np.ndarray:
        ImageTransforms._check_img(img)
        return cv2.flip(img, 0)

    @staticmethod
    def gaussian_blur(img: np.ndarray, ksize: int = 5, sigmaX: float = 0.0, kernel=None, **kwargs) -> np.ndarray:
        ImageTransforms._check_img(img)
        # 兼容旧调用：如果传入了 kernel（int 或 tuple），优先使用它
        if kernel is not None:
            if isinstance(kernel, (list, tuple)) and len(kernel) > 0:
                k = int(kernel[0])
            else:
                k = int(kernel)
        else:
            k = ksize
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(img, (k, k), sigmaX)

    @staticmethod
    def canny(img: np.ndarray, low: int = 50, high: int = 150, aperture: int = 3) -> np.ndarray:
        ImageTransforms._check_img(img)
        gray = ImageTransforms.to_gray(img)
        return cv2.Canny(gray, low, high, apertureSize=aperture, L2gradient=True)


def hsv_range(img: np.ndarray, lower: Sequence[int] = (0, 0, 0), upper: Sequence[int] = (255, 255, 255),
              morph: bool = True, ksize: int = 5) -> np.ndarray:
    """兼容旧名：`hsv_range` -> 代理到 `hsv_range_mask`。"""
    return ImageTransforms.hsv_range_mask(img, lower=lower, upper=upper, morph=morph, ksize=ksize)


def hsv_range_limit(img: np.ndarray, lower: Sequence[int] = (0, 0, 0), upper: Sequence[int] = (255, 255, 255),
                    morph: bool = True, ksize: int = 5) -> np.ndarray:
    """兼容旧名：`hsv_range_limit` -> 代理到 `hsv_range_mask`。"""
    return ImageTransforms.hsv_range_mask(img, lower=lower, upper=upper, morph=morph, ksize=ksize)


def resize_img(img: np.ndarray, size: Tuple[int, int] = (320, 240), interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """兼容旧名：`resize_img` -> 代理到 `ImageTransforms.resize`。"""
    return ImageTransforms.resize(img, size=size, interpolation=interpolation)


def filter_gaussianblur(img: np.ndarray, kernel=None, ksize: int = 5, sigmaX: float = 0.0) -> np.ndarray:
    """兼容旧名：接收 `kernel`（可为整数或(tuple)），适配到 `gaussian_blur` 的 `ksize`。"""
    if kernel is not None:
        # kernel 可以是 int 或 (kx, ky)
        if isinstance(kernel, (list, tuple)) and len(kernel) > 0:
            k = int(kernel[0])
        else:
            k = int(kernel)
    else:
        k = ksize
    return ImageTransforms.gaussian_blur(img, ksize=k, sigmaX=sigmaX)


__all__ = ["ImageTransforms"]

# 兼容性：把模块级别的函数也绑定到类，保留 `ImageTransforms.*` 的旧调用方式
for _name in [
    'hsv_range_limit', 'resize_img', 'resize_scale', 'filter_gaussianblur', 'filter_blur', 'filter_medianblur',
    'filter_bilateral', 'filter_box', 'to_canny', 'contrast_clahe', 'contrast_hist_equalization',
    'mor_dilate', 'mor_erode', 'mor_open', 'mor_close', 'mor_gradient', 'mor_tophat', 'mor_blackhat', 'mor_thinning'
]:
    if _name in globals():
        setattr(ImageTransforms, _name, staticmethod(globals()[_name]))

# 确保常见兼容名被绑定为类属性
if 'hsv_range' in globals():
    setattr(ImageTransforms, 'hsv_range', staticmethod(globals()['hsv_range']))
if 'resize_img' in globals():
    setattr(ImageTransforms, 'resize_img', staticmethod(globals()['resize_img']))


def filter_blur(img: np.ndarray, kernel=None, ksize: int = 5, **kwargs) -> np.ndarray:
    """兼容旧名：`filter_blur` -> 使用简单均值模糊适配旧 API（接受 kernel 或 ksize）。"""
    # 兼容不同旧参数名：kernel, k
    if kernel is None:
        k = kwargs.get('k', None)
        if k is None:
            k = ksize
    else:
        if isinstance(kernel, (list, tuple)) and len(kernel) > 0:
            k = int(kernel[0])
        else:
            k = int(kernel)
    if k <= 0:
        k = 1
    if k % 2 == 0:
        k += 1
    return cv2.blur(img, (k, k))


def blur(img: np.ndarray, ksize: int = 5, kernel=None, **kwargs) -> np.ndarray:
    """向后兼容的 `blur` 简单包装，返回均值模糊结果。"""
    # 兼容关键字 `kernel` 或 `k`
    if kernel is None:
        k = kwargs.get('k', None) or ksize
    else:
        if isinstance(kernel, (list, tuple)) and len(kernel) > 0:
            k = int(kernel[0])
        else:
            k = int(kernel)
    if k <= 0:
        k = 1
    if k % 2 == 0:
        k += 1
    return cv2.blur(img, (k, k))

# 兼容绑定：确保 filter_blur 和 blur 可作为类方法使用
setattr(ImageTransforms, 'filter_blur', staticmethod(filter_blur))
setattr(ImageTransforms, 'blur', staticmethod(blur))


def median_blur(img: np.ndarray, kernel=5, **kwargs) -> np.ndarray:
    k = int(kernel) if kernel is not None else 5
    if k % 2 == 0:
        k += 1
    return cv2.medianBlur(img, k)


def filter_medianblur(img: np.ndarray, kernel=5, **kwargs) -> np.ndarray:
    return median_blur(img, kernel=kernel)


def bilateral(img: np.ndarray, d: int = 9, sigmaColor: float = 75, sigmaSpace: float = 75, **kwargs) -> np.ndarray:
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)


def box_filter(img: np.ndarray, kernel=5, **kwargs) -> np.ndarray:
    k = int(kernel)
    return cv2.boxFilter(img, -1, (k, k))


def clahe(img: np.ndarray, clip_limit: float = 2.0, tileGridSize: tuple = (8, 8), **kwargs) -> np.ndarray:
    if img is None:
        raise ValueError('img is None')
    # 支持传入 int 或 tuple
    if isinstance(tileGridSize, int):
        tileGridSize = (tileGridSize, tileGridSize)
    if img.ndim == 2:
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tileGridSize)
        return clahe_obj.apply(img)
    # color image: apply to L channel in LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tileGridSize)
    l2 = clahe_obj.apply(l)
    lab = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def hist_equalize(img: np.ndarray, **kwargs) -> np.ndarray:
    if img is None:
        raise ValueError('img is None')
    if img.ndim == 2:
        return cv2.equalizeHist(img)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.equalizeHist(y)
    ycrcb = cv2.merge((y, cr, cb))
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _make_kernel(k):
    k = int(k) if k is not None else 3
    if k <= 0:
        k = 1
    return np.ones((k, k), np.uint8)


def dilate(img: np.ndarray, kernel=3, iterations: int = 1, **kwargs) -> np.ndarray:
    return cv2.dilate(img, _make_kernel(kernel), iterations=iterations)


def erode(img: np.ndarray, kernel=3, iterations: int = 1, **kwargs) -> np.ndarray:
    return cv2.erode(img, _make_kernel(kernel), iterations=iterations)


def morphology_open(img: np.ndarray, kernel=3, **kwargs) -> np.ndarray:
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, _make_kernel(kernel))


def morphology_close(img: np.ndarray, kernel=3, **kwargs) -> np.ndarray:
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, _make_kernel(kernel))


def morphology_gradient(img: np.ndarray, kernel=3, **kwargs) -> np.ndarray:
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, _make_kernel(kernel))


def morphology_tophat(img: np.ndarray, kernel=3, **kwargs) -> np.ndarray:
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, _make_kernel(kernel))


def morphology_blackhat(img: np.ndarray, kernel=3, **kwargs) -> np.ndarray:
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, _make_kernel(kernel))


# 将这些函数绑定为类静态方法以兼容旧调用风格
for _f in [
    'median_blur', 'filter_medianblur', 'bilateral', 'box_filter', 'clahe', 'hist_equalize',
    'dilate', 'erode', 'morphology_open', 'morphology_close', 'morphology_gradient', 'morphology_tophat', 'morphology_blackhat'
]:
    if _f in globals():
        setattr(ImageTransforms, _f, staticmethod(globals()[_f]))

# 额外别名：filter_bilateral -> bilateral
if 'bilateral' in globals():
    setattr(ImageTransforms, 'filter_bilateral', staticmethod(globals()['bilateral']))
if 'box_filter' in globals():
    setattr(ImageTransforms, 'filter_box', staticmethod(globals()['box_filter']))
if 'canny' in globals():
    setattr(ImageTransforms, 'to_canny', staticmethod(globals()['canny']))

# 如果 canny 是类方法，则也绑定 to_canny
if hasattr(ImageTransforms, 'canny'):
    setattr(ImageTransforms, 'to_canny', staticmethod(ImageTransforms.canny))


def to_canny(img: np.ndarray, min: int = 50, max: int = 150, apertureSize: int = 3, **kwargs) -> np.ndarray:
    # 兼容旧参数名：min/max/apertureSize -> low/high/aperture
    low = min
    high = max
    aperture = apertureSize
    return ImageTransforms.canny(img, low=low, high=high, aperture=aperture)

# 覆盖或绑定 to_canny 为兼容包装
setattr(ImageTransforms, 'to_canny', staticmethod(to_canny))

# mor_* 别名兼容
alias_map = {
    'mor_dilate': 'dilate',
    'mor_erode': 'erode',
    'mor_open': 'morphology_open',
    'mor_close': 'morphology_close',
    'mor_gradient': 'morphology_gradient',
    'mor_tophat': 'morphology_tophat',
    'mor_blackhat': 'morphology_blackhat',
}
for alias, target in alias_map.items():
    if target in globals():
        setattr(ImageTransforms, alias, staticmethod(globals()[target]))

# contrast 兼容名
if 'clahe' in globals():
    setattr(ImageTransforms, 'contrast_clahe', staticmethod(globals()['clahe']))
if 'hist_equalize' in globals():
    setattr(ImageTransforms, 'contrast_hist_equalization', staticmethod(globals()['hist_equalize']))


