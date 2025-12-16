# 艺术风格迁移 (Artistic Style Transfer)

本项目基于 TensorFlow 实现了一个神经网络风格迁移应用，使用了 TensorFlow Hub 上的预训练 Fast Style Transfer 模型。即可以把一张图片的“艺术风格”迁移到另一张“内容图片”上。

## 安装与设置

1.  **环境配置**:
    本项目建议在虚拟环境中运行。
    ```bash
    cd src
    python -m venv .venv
    # 激活虚拟环境 (Windows)
    .venv\\Scripts\\activate
    # 安装依赖
    pip install tensorflow tensorflow-hub matplotlib numpy pillow
    ```

2.  **如何运行**:
    在 `src` 目录下运行主脚本：
    ```bash
    python -m style_transfer.main
    ```

    默认情况下，脚本会在 `src` 目录下寻找 `content.jpg` (内容图) 和 `style.jpg` (风格图)。你可以直接替换这两个文件来尝试不同的效果。

## 项目结构

-   `style_transfer/`: 包含源代码的包。
    -   `main.py`: 主程序入口。
    -   `utils.py`: 图像处理工具函数。
-   `.venv/`: Python 虚拟环境 (通常不包含在代码库中)。

## 依赖库

-   Python 3.x
-   TensorFlow 2.x
-   TensorFlow Hub
-   Matplotlib, NumPy, Pillow
