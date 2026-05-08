import cv2
import numpy as np
from pyzbar import pyzbar
from typing import List, Dict, Any


class QRDetector:
    """封装 pyzbar 二维码检测器，提供多码检测接口"""

    def __init__(self):
        pass  # pyzbar 不需要初始化检测器

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        检测图像中的所有二维码，返回解码成功的二维码列表
        :param frame: BGR 图像
        :return: 每个二维码包含 'content', 'points', 'index' 等字段
        """
        # pyzbar 可以直接处理彩色图像，但灰度图通常效果更好
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用 pyzbar 检测二维码
        barcodes = pyzbar.decode(gray)

        detected = []
        for i, barcode in enumerate(barcodes):
            # 只处理二维码类型
            if barcode.type == 'QRCODE':
                # 获取二维码内容
                content = barcode.data.decode('utf-8')
                
                # 获取二维码四个角点（pyzbar 返回的是矩形边界框，需要转换为四点）
                # barcode.rect 包含 (x, y, width, height)
                # 我们需要转换为四个角点
                x, y, w, h = barcode.rect
                points = np.array([
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ], dtype=np.int32)
                
                detected.append({
                    'content': content,
                    'points': points,
                    'index': i
                })
        
        return detected