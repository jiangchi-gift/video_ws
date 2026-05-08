import json
from datetime import datetime
from typing import Dict, List


def save_data(qr_database: Dict, recognition_log: List[Dict], output_prefix: str = None):
    """
    将二维码数据库和识别日志保存为两个 JSON 文件。
    :param qr_database: key=content, value={'id', 'content', ...}
    :param recognition_log: 每帧的识别结果列表
    :param output_prefix: 文件名前缀，默认使用时间戳
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = output_prefix if output_prefix else ""

    # 简化数据库：只保留 id 和 content
    simplified_db = {}
    for content, data in qr_database.items():
        simplified_db[content] = {
            'id': data['id'],
            'content': data['content']
        }

    db_filename = f"{prefix}qr_database_{timestamp}.json"
    with open(db_filename, 'w', encoding='utf-8') as f:
        json.dump(simplified_db, f, ensure_ascii=False, indent=2)

    # 简化日志：只保留 frame、二维码 id、content 和物理坐标（保留4位小数）
    simplified_log = []
    for frame_data in recognition_log:
        sf = {'frame': frame_data['frame'], 'qr_codes': []}
        for qr in frame_data['qr_codes']:
            pos = [round(qr['position'][0], 4), round(qr['position'][1], 4), round(qr['position'][2], 4)]
            sf['qr_codes'].append({
                'id': qr['id'],
                'content': qr['content'],
                'position': pos
            })
        simplified_log.append(sf)

    log_filename = f"{prefix}recognition_log_{timestamp}.json"
    with open(log_filename, 'w', encoding='utf-8') as f:
        json.dump(simplified_log, f, ensure_ascii=False, indent=2)

    print(f"数据已保存到: {db_filename} 和 {log_filename}")

    # 实时输出当前帧的 JSON
    if simplified_log:
        print("当前帧识别结果:")
        print(json.dumps(simplified_log[-1], ensure_ascii=False, indent=2))