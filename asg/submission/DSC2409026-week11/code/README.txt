# Artistic Style Transfer

This project implements a neural style transfer application using TensorFlow and a pre-trained Fast Style Transfer model from TensorFlow Hub. It allows you to apply the artistic style of one image to another content image.

## Setup

1.  **Environment**: 
    Install the required dependencies directly:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Usage**:
    Run the main script:
    ```bash
    python -m style_transfer.main
    ```

    By default, the script looks for `content.jpg` and `style.jpg` in the current directory. You can replace these files with your own images.

## Project Structure

-   `style_transfer/`: Package containing source code.
    -   `main.py`: Main application script.
    -   `utils.py`: Utility functions for image processing.

## Requirements

-   Python 3.x
-   TensorFlow 2.x
-   TensorFlow Hub
-   Matplotlib, NumPy, Pillow