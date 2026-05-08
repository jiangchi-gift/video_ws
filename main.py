import cv2
from qr_detector import QRDetector
from qr_manager import QRManager
from camera_utils import create_default_camera_params
from data_saver import save_data


def main():
    # 加载相机参数（可以替换为自定义参数）
    camera_matrix, dist_coeffs, R, t, depth = create_default_camera_params()

    # 初始化模块
    detector = QRDetector()
    manager = QRManager(camera_matrix, dist_coeffs, R, t, depth)

    # 打开视频
    cap = cv2.VideoCapture('test1.mp4')
    if not cap.isOpened():
        print("无法打开视频文件 test1.mp4")
        return

    print("二维码追踪系统启动...")
    print("按 'q' 退出，按 's' 保存数据")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 检测二维码
        detected_qr_codes = detector.detect(frame)

        # 2. 处理数据（排序、分配ID、世界坐标转换、记录日志）
        frame_result, qr_with_ids = manager.process_frame_qr_data(detected_qr_codes)

        # 3. 可视化：在帧上绘制二维码边框和 ID
        for qr in qr_with_ids:
            pts = qr['points'].reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            x, y = pts[0][0]
            display_text = f"ID:{qr['id']} - {qr['content']}"
            cv2.putText(frame, display_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 4. 显示
        cv2.imshow("QR Code Tracking System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_data(manager.qr_database, manager.recognition_log)

    # 程序结束自动保存
    save_data(manager.qr_database, manager.recognition_log)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()