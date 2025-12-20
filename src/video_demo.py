
import tensorflow as tf
import cv2
import time
import argparse
import os
import sys
import numpy as np

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.style_transfer.model import make_style_transfer_network

def process_video(model_path, input_video, output_video, width=None):
    print("Initializing...")
    
    # 1. Load Model
    # Dynamic shape input
    transformer = make_style_transfer_network(input_shape=(None, None, 3))
    # Build dummy
    transformer(tf.zeros((1, 256, 256, 3)))
    
    print(f"Loading weights from {model_path}...")
    transformer.load_weights(model_path)
    
    # Compilation for speed (Crucial for Video)
    # We create a concrete function for the specific input size once we know it, 
    # OR we use a dynamic shape function. 
    # For consistent speed, if video size is constant, concrete function is best.
    
    @tf.function
    def style_transform(content_image):
        return transformer(content_image, training=False)

    # 2. Open Video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error opening video file {input_video}")
        return

    # Video properties
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Target dimensions
    if width:
        scale = width / orig_width
        new_width = width
        new_height = int(orig_height * scale)
    else:
        new_width = orig_width
        new_height = orig_height
        
    print(f"Processing Video: {orig_width}x{orig_height} -> {new_width}x{new_height} @ {fps} FPS")

    # Output Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (new_width, new_height))
    
    frame_count = 0
    total_inference_time = 0
    
    print("Starting processing loop...")
    
    # Warmup
    dummy = tf.zeros((1, new_height, new_width, 3), dtype=tf.float32)
    _ = style_transform(dummy)
    print("Warmup complete.")

    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        t0 = time.time()
        
        # Preprocess
        if width:
            frame = cv2.resize(frame, (new_width, new_height))
            
        # CV2 (BGR) -> RGB -> Tensor
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.float32)
        img_tensor = tf.expand_dims(img_tensor, 0) # (1, H, W, 3)
        
        # Inference (Input is 0-255)
        output_tensor = style_transform(img_tensor)
        
        # Postprocess
        # Clip and Cast
        output_tensor = tf.clip_by_value(output_tensor, 0.0, 255.0)
        output_img = output_tensor[0].numpy().astype(np.uint8)
        
        # RGB -> BGR
        output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        
        inference_time = time.time() - t0
        total_inference_time += inference_time
        frame_count += 1
        
        # Write
        out.write(output_bgr)
        
        # Console Logging
        if frame_count % 10 == 0:
            print(f"Frame {frame_count}: {1.0/inference_time:.1f} FPS (Inference)")
            
    total_time = time.time() - start_time
    avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
    
    cap.release()
    out.release()
    
    print(f"==========================================")
    print(f"Conversion Complete!")
    print(f"Processed {frame_count} frames in {total_time:.2f}s")
    print(f"Average Inference FPS: {avg_fps:.2f}")
    print(f"Output saved to: {output_video}")
    print(f"==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("input_video")
    parser.add_argument("--output", default="output_video.mp4")
    parser.add_argument("--width", type=int, default=None, help="Resize input to this width for speed")
    
    args = parser.parse_args()
    
    process_video(args.model_path, args.input_video, args.output, args.width)
