# Week 11 Submission - Group Project Phase 1

## Overview
This folder contains the **Phase 1 (MVP)** implementation of our "Artistic Style Transfer" project.
The code demonstrates the core functionality of style transfer using a pre-trained model for immediate validation.

## Roadmap to Final Submission
We are currently in **Phase 2**, which involves:
1.  **Custom Model Training**: We are preparing the COCO 2017 dataset to train our own Image Transformation Network (Johnson et al. architecture) from scratch using TensorFlow 2.x.
2.  **Web Interface**: We are developing an HTML5/Flask web interface to allow user-friendly interaction with the model.

## Folder Structure
-   `style_transfer/`: Core Python package for style transfer logic.
-   `main.py`: Entry point for the MVP.
-   `utils.py`: Image pre-processing utilities.

## How to Run (MVP)
(Ensure `tensorflow`, `tensorflow_hub`, `pillow` are installed)
```bash
python -m style_transfer.main
```
This will generate a `stylized_image.png` using the sample images.
