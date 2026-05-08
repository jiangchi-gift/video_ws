from typing import List, Dict, Any, Tuple
import numpy as np
from datetime import datetime
from camera_utils import pixel_to_world


class QRManager:
    """
    管理二维码数据库、排序、ID 分配及世界坐标转换。
    相机内外参通过构造函数注入。
    """

    def __init__(self, camera_matrix, dist_coeffs, R, t, depth):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.R = R
        self.t = t
        self.depth = depth

        self.qr_database: Dict[str, Dict] = {}   # key = content, value = {'id', 'content', ...}
        self.frame_count = 0
        self.recognition_log: List[Dict] = []
        
        # 稳定性参数
        self.detection_history: Dict[str, int] = {}  # 记录每个二维码的连续检测次数
        self.min_detection_frames = 2  # 至少连续检测2帧才显示
        self.max_miss_frames = 3  # 最多连续丢失3帧才移除
        self.y_buffer = 30  # 上下排分界线缓冲区（像素）

    @staticmethod
    def calculate_center(points: np.ndarray) -> Tuple[float, float]:
        """计算二维码四个角点的中心点（像素坐标）"""
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        return center_x, center_y

    def assign_ids_and_sort(self, detected_qr_codes: List[Dict]) -> List[Dict]:
        """
        根据二维码在图像中的上下位置自动分配 ID：
        上面一排（较小的 y）从左到右分配 ID 1-3，
        下面一排（较大的 y）从左到右分配 ID 4-6。
        添加稳定性过滤：只有连续检测到的二维码才显示，避免闪烁。
        """
        # 计算中心点
        for qr in detected_qr_codes:
            qr['center'] = self.calculate_center(qr['points'])

        # 更新检测历史
        detected_contents = {qr['content'] for qr in detected_qr_codes}
        
        # 增加当前帧检测到的二维码的计数
        for content in detected_contents:
            self.detection_history[content] = self.detection_history.get(content, 0) + 1
        
        # 减少未检测到的二维码的计数
        contents_to_remove = []
        for content in self.detection_history:
            if content not in detected_contents:
                self.detection_history[content] -= 1
                if self.detection_history[content] <= -self.max_miss_frames:
                    contents_to_remove.append(content)
        
        # 移除连续丢失过多帧的二维码
        for content in contents_to_remove:
            del self.detection_history[content]

        # 过滤不稳定的检测结果（只保留连续检测到的）
        stable_qr_codes = [
            qr for qr in detected_qr_codes
            if self.detection_history.get(qr['content'], 0) >= self.min_detection_frames
        ]

        if not stable_qr_codes:
            return []

        # 上下分组（使用缓冲区避免边界波动）
        y_coords = [qr['center'][1] for qr in stable_qr_codes]
        y_min, y_max = min(y_coords), max(y_coords)
        y_median = (y_min + y_max) / 2

        top_row, bottom_row = [], []
        for qr in stable_qr_codes:
            # 检查是否已存在于数据库中（保持ID稳定）
            if qr['content'] in self.qr_database:
                existing_id = self.qr_database[qr['content']]['id']
                # 根据已有ID判断归属排
                if existing_id <= 3:
                    top_row.append(qr)
                else:
                    bottom_row.append(qr)
            else:
                # 新二维码：使用带缓冲区的分界线
                if qr['center'][1] <= y_median - self.y_buffer:
                    top_row.append(qr)
                elif qr['center'][1] >= y_median + self.y_buffer:
                    bottom_row.append(qr)
                else:
                    # 在缓冲区范围内，暂不分配，等待更稳定的检测
                    continue

        # 每组内从左到右排序
        top_row.sort(key=lambda q: q['center'][0])
        bottom_row.sort(key=lambda q: q['center'][0])

        # 分配 ID
        for i, qr in enumerate(top_row):
            content = qr['content']
            if content in self.qr_database:
                qr['id'] = self.qr_database[content]['id']
            else:
                qr['id'] = i + 1
                self.qr_database[content] = {
                    'id': qr['id'],
                    'content': content,
                    'first_detected': datetime.now().isoformat(),
                    'detection_count': 1
                }

        for i, qr in enumerate(bottom_row):
            content = qr['content']
            if content in self.qr_database:
                qr['id'] = self.qr_database[content]['id']
            else:
                qr['id'] = i + 4
                self.qr_database[content] = {
                    'id': qr['id'],
                    'content': content,
                    'first_detected': datetime.now().isoformat(),
                    'detection_count': 1
                }

        # 更新检测次数
        for qr in top_row + bottom_row:
            if qr['content'] in self.qr_database:
                self.qr_database[qr['content']]['detection_count'] += 1

        return top_row + bottom_row

    def get_world_coordinates(self, qr: Dict) -> Tuple[float, float, float]:
        """获取二维码中心点的世界坐标"""
        center_x, center_y = qr['center']
        return pixel_to_world(
            center_x, center_y,
            self.camera_matrix, self.dist_coeffs,
            self.R, self.t, self.depth
        )

    def process_frame_qr_data(self, detected_qr_codes: List[Dict]) -> dict:
        """
        处理当前帧的二维码数据：排序、分配 ID、转换世界坐标，
        并记录到日志中。返回当前帧的结果字典（用于保存）。
        """
        qr_with_ids = self.assign_ids_and_sort(detected_qr_codes)

        frame_result = {
            'frame': self.frame_count,
            'timestamp': datetime.now().isoformat(),
            'qr_codes': []
        }

        for qr in qr_with_ids:
            world_x, world_y, world_z = self.get_world_coordinates(qr)
            frame_result['qr_codes'].append({
                'id': qr['id'],
                'content': qr['content'],
                'position': [world_x, world_y, world_z],
                'pixel_position': qr['center']
            })

        self.recognition_log.append(frame_result)
        self.frame_count += 1
        return frame_result, qr_with_ids