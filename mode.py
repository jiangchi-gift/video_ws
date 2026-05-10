import cv2
import numpy as np
from abc import ABC, abstractmethod

# =========================
# 抽象基类
# =========================
class get_image_base(ABC):

    @abstractmethod
    def get_image(self):
        pass


# =========================
# 摄像头获取
# =========================
class use_camera(get_image_base):

    def __init__(self, num=0, width=640, height=480):
        self.cap = cv2.VideoCapture(num)

        if not self.cap.isOpened():
            print("无法打开摄像头")
            exit()

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get_image(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame


# =========================
# 视频文件读取
# =========================
class use_video(get_image_base):

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            print("视频打不开！")
            exit()

    def get_image(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame


# =========================
# 图像识别封装（核心）
# =========================
class RedDetector:

    def __init__(self):
        self.lower_red1 = np.array([0, 120, 70])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 70])
        self.upper_red2 = np.array([180, 255, 255])

    def detect(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = mask1 + mask2

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        result = frame.copy()
        boxes = []

        for cnt in contours:
            if cv2.contourArea(cnt) < 500:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2

            boxes.append((x, y, w, h, cx, cy))

            # 画框
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # 中心点
            cv2.circle(result, (cx, cy), 5, (0, 255, 0), -1)

            # 显示信息
            text = f"x:{x} y:{y} cx:{cx} cy:{cy}"
            cv2.putText(result, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return result, mask, boxes


# =========================
# 主程序（直接运行用）
# =========================
if __name__ == "__main__":

    # 这里改成你的视频名字
    source = use_video("01.mp4")
    # source = use_camera(0)   # 如果用摄像头就打开这个

    detector = RedDetector()

    frame_id = 0

    while True:
        frame = source.get_image()

        if frame is None:
            print("结束")
            break

        frame_id += 1

        result, mask, boxes = detector.detect(frame)

        # 显示帧数
        cv2.putText(result, f"Frame:{frame_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Result", result)
        cv2.imshow("Mask", mask)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()