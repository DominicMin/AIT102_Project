# AIT102 风格迁移演示系统 - 技术实施指南

## 1. 项目背景与目标 (Context)
本项目旨在利用 **Dual RTX 3090 (48GB VRAM)** 服务器的强大算力，构建一个用于 Presentation 的高性能风格迁移演示系统。
**核心卖点**：
-   **并行风格矩阵**：利用双卡并发能力，一次生成多张不同风格图片。
-   **God Mode (实时视频)**：利用 WebRTC 实现低延迟实时视频风格迁移。
-   **零延迟体验**：模型全量预加载，无冷启动时间。

---

## 2. 后端开发指南 (Backend & AI)
**负责人**：炼丹师 & 后端工程师
**技术栈**：Python 3.9, TensorFlow 2.14, FastAPI, WebRTC

### 2.1 模型训练 (Model Training)
我们需要训练 5 个独立的 Transformer 模型。
*   **基础代码**：`src/train_v2.py` (已支持 MirroredStrategy 并行训练)。
*   **目标风格**：
    1.  `style_picasso.jpg` (毕加索 - 立体主义)
    2.  `style_vangogh.jpg` (梵高 - 星空)
    3.  `style_ukiyoe.jpg` (浮世绘 - 海浪)
    4.  `style_cyberpunk.jpg` (赛博朋克 - 强光/霓虹)
    5.  `style_sketch.jpg` (素描 - 线条)
*   **训练参数**：
    *   `batch_size`: 32 (并行加速)
    *   `epochs`: 40 (保证高质量)
    *   `image_size`: 256x256 (训练用)，推理支持任意尺寸。
*   **产出物**：5 个 `.h5` 或 `SavedModel` 权重文件，存放在 `models/` 目录下。

### 2.2 推理服务 (Inference Server)
使用 `FastAPI` 构建服务，核心逻辑如下：
1.  **启动加载 (Warm-up)**：
    *   服务启动时，读取所有 5 个模型权重。
    *   **显存分配策略**：
        *   GPU 0: 加载 Model 1, 2, 3
        *   GPU 1: 加载 Model 4, 5
        *   *注：RTX 3090 显存足够放下这些轻量级 Transformer。*
2.  **并行处理 (Parallel matrix)**：
    *   当收到 `/transform_all` 请求时，使用 `concurrent.futures.ThreadPoolExecutor` 同时向两个 GPU 提交推理任务。
3.  **实时流 (WebRTC)**：
    *   使用 `aiortc` 库处理 WebRTC 信令。
    *   建立视频通道后，帧处理回调函数中调用指定的风格模型进行 `predict`。

### 2.3 API 接口规范 (API Spec)

#### `GET /styles`
返回可用风格列表。
```json
{
  "styles": [
    {"id": "picasso", "name": "Picasso", "preview_url": "/static/previews/picasso.jpg"},
    {"id": "vangogh", "name": "Van Gogh", "preview_url": "/static/previews/vangogh.jpg"},
    ...
  ]
}
```

#### `POST /transform` (单图单风格)
*   **Form Data**: `file` (Image), `style_id` (String)
*   **Response**: JPEG Image bytes

#### `POST /transform_all` (九宫格模式)
*   **Form Data**: `file` (Image)
*   **Response**: JSON list of base64 images (or URLs)
```json
{
  "results": [
    {"style_id": "picasso", "image_base64": "..."},
    {"style_id": "vangogh", "image_base64": "..."}
  ]
}
```

---

## 3. 前端开发指南 (Frontend)
**负责人**：前端工程师
**技术栈**：Next.js, TailwindCSS (Dark Mode), Framer Motion (动画)

### 3.1 核心页面设计
1.  **Hero Section**:
    *   极简设计，大标题 + 拖拽上传区域。
    *   背景可以是动态的流体渐变。
2.  **Results Grid (矩阵展示)**:
    *   一旦上传图片，触发 `/transform_all`。
    *   **动画效果**：中心原图“炸裂”散开，变成周围环绕的 5 张风格图。
    *   支持点击任意风格图放大，并提供 "Original vs Styled" 拖拽对比条。
3.  **God Mode (视频入口)**:
    *   一个醒目的 "Go Live" 按钮。
    *   点击后全屏，调用浏览器摄像头 API (`navigator.mediaDevices.getUserMedia`)。
    *   建立 WebRTC 连接，显示处理后的视频流。

### 3.2 性能优化
*   图片上传前进行客户端压缩 (最大 2048px)，避免 4K原图直接上传导致的非必要网络延迟（除非开启 HD Mode）。
*   WebRTC 需处理丢包重连逻辑。

---

## 4. 联调与部署 (Deployment)
*   **服务器环境**：Ubuntu 22.04, Conda `ait` 环境。
*   **端口映射**：FastAPI (8000) -> Next.js (3000)。
*   **局域网访问**：确保 Presentation 笔记本和服务器在同一 LAN 下（Cyberjaya 网络），通过内网 IP 访问以获得最低延迟。
