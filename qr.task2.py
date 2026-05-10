import cv2
import pyzbar.pyzbar as pyzbar
import json
import os

qr_data = {}

def detect_qr_codes(frame):
    qr_list = []
    # 轻便增强，不卡顿且识别力极强
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 二值化是识别后排二维码的关键
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    decoded_objects = pyzbar.decode(binary)

    for obj in decoded_objects:
        data = obj.data.decode("utf-8")
        rect = obj.rect
        center_x = rect.left + rect.width // 2
        center_y = rect.top + rect.height // 2

        # 去重
        repeat = False
        for item in qr_list:
            if item["data"] == data:
                repeat = True
                break
        if not repeat:
            qr_list.append({
                "data": data,
                "center_x": center_x,
                "center_y": center_y,
                "rect": rect
            })
    return qr_list

def sort_qr_codes(qr_list):
    if not qr_list:
        return []
    qr_list.sort(key=lambda x: x["center_y"])
    mid_y = (qr_list[0]["center_y"] + qr_list[-1]["center_y"]) / 2
    upper_row = [q for q in qr_list if q["center_y"] < mid_y]
    lower_row = [q for q in qr_list if q["center_y"] >= mid_y]
    upper_row.sort(key=lambda x: x["center_x"])
    lower_row.sort(key=lambda x: x["center_x"])
    return upper_row + lower_row

def save_qr_data(sorted_qr):
    global qr_data
    qr_data = {}
    for idx, qr in enumerate(sorted_qr, start=1):
        qr_data[str(idx)] = {
            "content": qr["data"],
            "center_x": qr["center_x"],
            "center_y": qr["center_y"]
        }
    with open("qr_records.json", "w", encoding="utf-8") as f:
        json.dump(qr_data, f, ensure_ascii=False, indent=4)
    print("全部二维码识别完成并保存")

def check_qr_and_show(frame, qr_list):
    for qr in qr_list:
        for qr_id, info in qr_data.items():
            if info["content"] == qr["data"]:
                rect = qr["rect"]
                cv2.rectangle(frame, (rect.left, rect.top),
                              (rect.left + rect.width, rect.top + rect.height),
                              (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{qr_id}", (rect.left, rect.top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                break

def main():
    video_path = r"C:\Users\Cooor\OneDrive\Desktop\opencv-example\qr.recognition\QR.code_video.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("视频打开失败")
        return

    # 限速保证绝对流畅
    cap.set(cv2.CAP_PROP_FPS, 15)
    first_detect_done = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        qr_list = detect_qr_codes(frame)

        if not first_detect_done and len(qr_list) >= 6:
            sorted_qr = sort_qr_codes(qr_list)
            save_qr_data(sorted_qr)
            first_detect_done = True

        if first_detect_done:
            check_qr_and_show(frame, qr_list)

        cv2.imshow("QR", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
     