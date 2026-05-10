import cv2
import pyzbar.pyzbar as pyzbar
import json
import os

# 存储已识别的二维码信息
qr_data = {}

def detect_qr_codes(frame):
    """增强识别，解决后面二维码识别不出的问题"""
    # 灰度化+降噪，大幅提升识别率
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    decoded_objects = pyzbar.decode(blurred)
    qr_list = []
    for obj in decoded_objects:
        # 获取二维码内容
        data = obj.data.decode("utf-8")
        # 获取二维码中心坐标
        rect = obj.rect
        center_x = rect.left + rect.width // 2
        center_y = rect.top + rect.height // 2
        qr_list.append({
            "data": data,
            "center_x": center_x,
            "center_y": center_y,
            "rect": rect
        })
    return qr_list

def sort_qr_codes(qr_list):
    """按左上到右下排序（先按y从小到大，y相近按x从小到大）"""
    if not qr_list:
        return []
    # 先按y坐标排序，确定上下行
    qr_list.sort(key=lambda x: x["center_y"])
    # 取中间y值作为上下行分界
    mid_y = (qr_list[0]["center_y"] + qr_list[-1]["center_y"]) / 2
    upper_row = [q for q in qr_list if q["center_y"] < mid_y]
    lower_row = [q for q in qr_list if q["center_y"] >= mid_y]
    # 每行内按x从左到右排序
    upper_row.sort(key=lambda x: x["center_x"])
    lower_row.sort(key=lambda x: x["center_x"])
    # 合并成左上到右下的顺序
    sorted_qr = upper_row + lower_row
    return sorted_qr

def save_qr_data(sorted_qr):
    """保存识别结果为JSON"""
    global qr_data
    # 按顺序分配ID（1~6）
    qr_data = {}
    for idx, qr in enumerate(sorted_qr, start=1):
        qr_data[str(idx)] = {
            "content": qr["data"],
            "center_x": qr["center_x"],
            "center_y": qr["center_y"]
        }
    # 保存到文件
    with open("qr_records.json", "w", encoding="utf-8") as f:
        json.dump(qr_data, f, ensure_ascii=False, indent=4)
    print("识别结果已保存到 qr_records.json")
    print("JSON输出:")
    print(json.dumps(qr_data, ensure_ascii=False, indent=4))

def check_qr_and_show(frame, qr_list):
    """检查二维码是否已保存，已保存则弹出信息"""
    for qr in qr_list:
        # 遍历已保存的数据，匹配内容
        for qr_id, info in qr_data.items():
            if info["content"] == qr["data"]:
                # 在画面上绘制信息
                rect = qr["rect"]
                cv2.rectangle(frame, (rect.left, rect.top), 
                              (rect.left + rect.width, rect.top + rect.height), 
                              (0, 255, 0), 2)
                text = f"ID: {qr_id}, Content: {info['content']}"
                cv2.putText(frame, text, (rect.left, rect.top - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                break

def main():
    # 视频的完整路径
    video_path = r"C:\Users\Cooor\OneDrive\Desktop\opencv-example\qr.recognition\QR.code_video.mp4"
    
    print("当前工作目录:", os.getcwd())
    print("视频文件路径:", video_path)
    print("视频文件是否存在:", os.path.exists(video_path))

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件，请检查：")
        print("1. 视频文件路径是否正确")
        print("2. 视频文件是否损坏")
        return
    
    # 标记是否已完成首次识别
    first_detect_done = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频播放完毕")
            break
        
        # 识别二维码
        qr_list = detect_qr_codes(frame)
        
        if not first_detect_done and qr_list:
            # 首次识别并排序保存
            sorted_qr = sort_qr_codes(qr_list)
            save_qr_data(sorted_qr)
            first_detect_done = True
        
        if first_detect_done and qr_list:
            # 后续识别时弹出信息
            check_qr_and_show(frame, qr_list)
        
        # 显示画面
        cv2.imshow("QR Code Tracking", frame)
        
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()