
import cv2
import time
import argparse
import os
import sys
import numpy as np

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf

# Ensure local imports work if running from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.style_transfer.model import make_style_transfer_network
except ImportError:
    # Fallback if running directly inside src/
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from style_transfer.model import make_style_transfer_network

def run_local_demo(model_path, camera_id=0, width=640, window_name="AIT102 Style Transfer (God Mode)"):
    print("=========================================")
    print(f"Initializing Local Demo...")
    print(f"Target Width: {width}")
    print(f"Device: {tf.config.list_physical_devices('GPU')}")
    print("=========================================")

    # 1. Load Model
    print(f"Loading Model: {model_path}")
    try:
        transformer = make_style_transfer_network(input_shape=(None, None, 3))
        # Warmup / Init
        transformer(tf.zeros((1, 256, 256, 3)))
        transformer.load_weights(model_path)
        print("Model Loaded Successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Optimization: tf.function for speed
    @tf.function
    def predict(input_tensor):
        return transformer(input_tensor, training=False)

    # 2. Open Camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    # Limit resolution if acceptable (reduces USB bandwidth load)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Starting Main Loop. Press 'q' to quit.")
    
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Resize for inference speed / visual consistency
        h, w = frame.shape[:2]
        if width is not None and w != width:
            scale = width / w
            new_h = int(h * scale)
            frame = cv2.resize(frame, (width, new_h))
        
        # Preprocess
        # BGR -> RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.float32)
        img_tensor = tf.expand_dims(img_tensor, 0) # (1, H, W, 3)

        # Inference
        output_tensor = predict(img_tensor)

        # Postprocess
        output_tensor = tf.clip_by_value(output_tensor, 0.0, 255.0)
        output_img = output_tensor[0].numpy().astype(np.uint8)
        
        # RGB -> BGR
        start_display = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        # Draw FPS
        cv2.putText(start_display, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show result
        cv2.imshow(window_name, start_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to .h5 style model")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Inference width. Higher = Slower but clearer.")
    
    args = parser.parse_args()
    
    # On Windows, you might verify the path
    if not os.path.exists(args.model_path):
        print(f"Warning: Model path {args.model_path} does not exist!")
    
    run_local_demo(args.model_path, args.camera, args.width)
