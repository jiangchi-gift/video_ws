import json
import threading
import logging
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class QRManager:
    """管理二维码识别、ID 分配和世界坐标转换的工具类。

    特性：
    - 支持并发访问（内部锁）
    - 可配置的稳定性阈值与分配策略
    - 可持久化 QR 数据库到 JSON
    """

    def __init__(self, camera_matrix, dist_coeffs, R, t, depth,
                 min_detection_frames: int = 2, max_miss_frames: int = 3, y_buffer: int = 30,
                 persist_path: Optional[str] = None):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.R = R
        self.t = t
        self.depth = depth

        self.qr_database: Dict[str, Dict] = {}
        self.frame_count = 0
        self.recognition_log: List[Dict] = []
        self.detection_history: Dict[str, int] = {}
        self.min_detection_frames = int(min_detection_frames)
        self.max_miss_frames = int(max_miss_frames)
        self.y_buffer = int(y_buffer)
        self._lock = threading.RLock()
        self.persist_path = persist_path
        if persist_path:
            try:
                self._load_db()
            except Exception:
                logger.exception("加载持久化 QR 数据库失败，继续使用空数据库")

    def _load_db(self):
        with open(self.persist_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.qr_database = data.get('qr_database', {})

    def _save_db(self):
        if not self.persist_path:
            return
        with open(self.persist_path, 'w', encoding='utf-8') as f:
            json.dump({'qr_database': self.qr_database}, f, ensure_ascii=False, indent=2)

    @staticmethod
    def calculate_center(points: np.ndarray) -> Tuple[float, float]:
        pts = np.asarray(points)
        if pts.size == 0:
            return 0.0, 0.0
        x_coords = pts[:, 0]
        y_coords = pts[:, 1]
        return float(x_coords.mean()), float(y_coords.mean())

    def assign_ids_and_sort(self, detected_qr_codes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        with self._lock:
            for qr in detected_qr_codes:
                qr['center'] = self.calculate_center(qr['points'])

            detected_contents = {qr['content'] for qr in detected_qr_codes}
            for content in detected_contents:
                self.detection_history[content] = self.detection_history.get(content, 0) + 1

            # Decrease counters for missing ones
            for content in list(self.detection_history.keys()):
                if content not in detected_contents:
                    self.detection_history[content] -= 1
                    if self.detection_history[content] <= -self.max_miss_frames:
                        del self.detection_history[content]

            stable_qr_codes = [qr for qr in detected_qr_codes if self.detection_history.get(qr['content'], 0) >= self.min_detection_frames]
            if not stable_qr_codes:
                return []

            y_coords = [qr['center'][1] for qr in stable_qr_codes]
            y_min, y_max = min(y_coords), max(y_coords)
            y_median = (y_min + y_max) / 2

            top_row, bottom_row = [], []
            for qr in stable_qr_codes:
                content = qr['content']
                if content in self.qr_database:
                    existing_id = self.qr_database[content]['id']
                    if existing_id <= 3:
                        top_row.append(qr)
                    else:
                        bottom_row.append(qr)
                else:
                    cy = qr['center'][1]
                    if cy <= y_median - self.y_buffer:
                        top_row.append(qr)
                    elif cy >= y_median + self.y_buffer:
                        bottom_row.append(qr)
                    else:
                        # 在缓冲区内，跳过本次分配
                        continue

            top_row.sort(key=lambda q: q['center'][0])
            bottom_row.sort(key=lambda q: q['center'][0])

            # 分配 ID（可定制规则，这里保持原始逻辑）
            for i, qr in enumerate(top_row):
                content = qr['content']
                if content in self.qr_database:
                    qr['id'] = self.qr_database[content]['id']
                else:
                    qr['id'] = i + 1
                    self.qr_database[content] = {'id': qr['id'], 'content': content, 'first_detected': datetime.now().isoformat(), 'detection_count': 1}

            for i, qr in enumerate(bottom_row):
                content = qr['content']
                if content in self.qr_database:
                    qr['id'] = self.qr_database[content]['id']
                else:
                    qr['id'] = i + 4
                    self.qr_database[content] = {'id': qr['id'], 'content': content, 'first_detected': datetime.now().isoformat(), 'detection_count': 1}

            for qr in top_row + bottom_row:
                if qr['content'] in self.qr_database:
                    self.qr_database[qr['content']]['detection_count'] = self.qr_database[qr['content']].get('detection_count', 0) + 1

            # 尝试持久化
            try:
                self._save_db()
            except Exception:
                logger.exception("持久化 QR 数据库失败")

            return top_row + bottom_row

    def get_world_coordinates(self, qr: Dict[str, Any]) -> Tuple[float, float, float]:
        # 延迟导入以避免循环依赖
        try:
            import pkg
            camera_utils = getattr(pkg, 'camera_utils', None)
            if camera_utils is None:
                from pkg import _3_camera_utils as camera_utils
        except Exception:
            # 退回到局部导入
            from pkg import _3_camera_utils as camera_utils

        center_x, center_y = qr['center']
        return camera_utils.pixel_to_world(center_x, center_y, self.camera_matrix, self.dist_coeffs, self.R, self.t, self.depth)

    def process_frame_qr_data(self, detected_qr_codes: List[Dict[str, Any]]) -> Tuple[dict, List[Dict[str, Any]]]:
        with self._lock:
            qr_with_ids = self.assign_ids_and_sort(detected_qr_codes)
            frame_result = {'frame': self.frame_count, 'timestamp': datetime.now().isoformat(), 'qr_codes': []}
            for qr in qr_with_ids:
                try:
                    world_x, world_y, world_z = self.get_world_coordinates(qr)
                except Exception:
                    logger.exception("世界坐标转换失败")
                    world_x = world_y = world_z = 0.0
                frame_result['qr_codes'].append({'id': qr.get('id'), 'content': qr.get('content'), 'position': [world_x, world_y, world_z], 'pixel_position': qr.get('center')})
            self.recognition_log.append(frame_result)
            self.frame_count += 1
            return frame_result, qr_with_ids


__all__ = ["QRManager"]


def load_qr_db(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('qr_database', {})


def save_qr_db(path: str, qr_database: dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'qr_database': qr_database}, f, ensure_ascii=False, indent=2)


# 绑定实例方法的兼容别名
setattr(QRManager, 'save_db', QRManager._save_db)
setattr(QRManager, 'load_db', QRManager._load_db)

__all__.extend(["load_qr_db", "save_qr_db"])
