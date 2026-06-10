# 项目说明：`pkg` 封装库（详细）

本仓库将多个与图像处理、摄像头接口和二维码识别相关的模块，统一封装为本地包 `pkg`。目标是：

- 降低上游项目改动成本（通过兼容别名与包装函数）
- 提供更稳健的摄像头与图像处理基础组件
- 支持在不同机器/环境中复用（含依赖回退策略）

## 模块映射与动态加载

`pkg` 的子模块由目录下的文件按名称自动注册（去掉前缀数字）：

- `pkg/1_image_transforms.py` -> `pkg.image_transforms`
- `pkg/2_camera.py` -> `pkg.camera`
- `pkg/3_camera_utils.py` -> `pkg.camera_utils`
- `pkg/4_hsv.py` -> `pkg.hsv`
- `pkg/5_qr_detector.py` -> `pkg.qr_detector`
- `pkg/6_qr_manager.py` -> `pkg.qr_manager`

实现入口： [pkg/__init__.py](pkg/__init__.py)

## 推荐安装流程

使用虚拟环境（venv 或 conda）：

```bash
# Windows（PowerShell）
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

安装主要依赖：

```bash
pip install opencv-python numpy
# 可选：
pip install pyzbar
```

生成 `requirements.txt`（可选、用于部署）：

```bash
pip freeze > requirements.txt
```

最小依赖清单（示例）：

```
opencv-python
numpy
pyzbar  # 可选
```

## API 快速参考（常用）

下面列出最常用的类与函数，示例以 Python 风格展示。完整签名与实现请查看源码文件。

### `pkg.image_transforms`（ImageTransforms）

- to_gray(img) -> gray
- to_hsv(img) -> hsv
- hsv_range_mask(img, lower, upper, morph=True, ksize=5) -> mask
- resize(img, (w,h)) 或 resize_scale(img, scale)
- rotate(img, angle)
- gaussian_blur(img, ksize=5), blur, median_blur, bilateral
- canny(img, t1, t2)
- morphology: dilate/erode/open/close/gradient/tophat/blackhat
- 兼容别名：`resize_img`, `filter_gaussianblur`, `to_canny` 等

示例：

```py
from pkg import image_transforms as it
img = cv2.imread('input.jpg')
mask = it.ImageTransforms.hsv_range_mask(img, (50,50,50), (70,255,255))
```

### `pkg.camera`（Camera）

- 构造：`Camera(camera_id=0, backend=cv2.CAP_DSHOW, reconnect_attempts=3)`
- `read(timeout=2.0)` -> (frame, timestamp) 或 None
- `read_frame(timeout=2.0)` -> frame 或 None（向后兼容）
- `release()`, `is_opened()`, `set_resolution(w,h)`
- 兼容别名：`open_camera()` 返回 `Camera` 实例；`Camera.readFrame` 为别名

示例：

```py
from pkg.camera import open_camera
cam = open_camera(0)
frame = cam.read_frame()
cam.release()
```

### `pkg.camera_utils`

- `create_default_camera_params()` -> (camera_matrix, dist_coeffs, R, t, depth)
- `pixel_to_world(x,y, camera_matrix, dist_coeffs, R, t, depth)` -> (Xw,Yw,Zw) 或 N x 3 数组
- 兼容：`pixel_to_world_xy`, `safe_create_default_camera_params`

示例：

```py
cm, dc, R, t, depth = pkg.camera_utils.create_default_camera_params()
Xw, Yw, Zw = pkg.camera_utils.pixel_to_world(100,200, cm, dc, R, t, depth)
```

### `pkg.hsv`

- `hsv_mask_from_image(img, lower, upper, morph=True, ksize=5)`
- `calibrate_from_image(path, save_to=None)`（交互窗口）
- `calibrate_from_camera()`（交互窗口）
- 兼容：`hsv_mask`, `save_thresholds`, `load_thresholds`, `calibrate_from_file`

示例：

```py
mask = pkg.hsv.hsv_mask_from_image(img, (0,50,50), (10,255,255))
```

### `pkg.qr_detector`（QRDetector）

- `QRDetector(use_pyzbar=True)` 如果系统已安装 `pyzbar` 则优先使用
- `detect(frame)` -> list of {content, points, method}
- 兼容函数：`detect_first(frame)` 返回首个结果或 None；`decode_pyzbar(frame)` 仅在 pyzbar 可用时直接返回原始解码

示例：

```py
det = pkg.qr_detector.QRDetector()
res = det.detect(frame)
for qr in res:
	print(qr['content'])
```

### `pkg.qr_manager`（QRManager）

- 构造：`QRManager(camera_matrix, dist_coeffs, R, t, depth, persist_path=None, ...)`
- `assign_ids_and_sort(detected_qr_codes)`
- `process_frame_qr_data(detected_qr_codes)` -> (frame_result, qr_with_ids)
- 持久化工具：`load_qr_db(path)`, `save_qr_db(path)`

示例：

```py
mgr = pkg.qr_manager.QRManager(cm, dc, R, t, depth, persist_path='qr_db.json')
frame_result, qr_list = mgr.process_frame_qr_data(detected)
```

## 示例脚本

推荐先运行以下示例以验证运行环境：

```bash
python img/example_usage.py
python Package/img/example_usage.py
python QR_code/main.py
```

注意：某些示例会尝试打开摄像头或读取视频文件（例如 `test1.mp4`），缺少资源时脚本会尽量使用占位图像或优雅退出。

## 打包与发布建议

- 如需发布为 pip 包，添加 `pyproject.toml` 或 `setup.cfg` 并使用 `python -m build`。
- 生成 wheel：

```bash
python -m build
```

在 CI 上做的建议步骤：

1. 创建虚拟环境并安装依赖
2. 运行静态检查（如 `flake8`）
3. 运行示例与单元测试
4. 构建 wheel 并上传到内部 PyPI（如果需要）

## 常见问题与排查

- pyzbar 缺失：安装 `pyzbar`，否则 `pkg.qr_detector` 会回退至 OpenCV 的 `QRCodeDetector`。

```bash
pip install pyzbar
```

- opencv-contrib：部分高级形态学/细化算法依赖 `opencv-contrib-python`，如出现 AttributeError 请安装：

```bash
pip install opencv-contrib-python
```

- 摄像头打不开：尝试 `pkg.camera.find_camera()` 检查可用设备或更换 backend（如 `cv2.CAP_MSMF`）。

## 版本控制与提交建议

提交示例：

```bash
git add .
git commit -m "chore(pkg): compatibility wrappers, update README"
```

我可以为你：

- 生成 `requirements.txt` 并保存到仓库；
- 将改动 commit 到本地 git（需要确认）；
- 生成打包配置（`pyproject.toml`）并构建 wheel。

---

如果希望我把 README 拆分为 `README_USER.md`（使用手册）和 `README_DEV.md`（开发者指南），或者需要英文版，请告诉我。