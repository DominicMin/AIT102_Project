
import cv2
import time
import argparse
import os
import sys
import numpy as np
import onnxruntime as ort

def run_local_demo_onnx(model_path, camera_id=0, width=640, window_name="AIT102 Style Transfer (ONNX God Mode)"):
    print("=========================================")
    print(f"Initializing ONNX Local Demo...")
    print(f"Target Width: {width}")
    
    # Check Providers (CUDA, DirectML, CPU)
    available_providers = ort.get_available_providers()
    print(f"Available Providers: {available_providers}")
    
    # Prioritize DirectML for easy Windows setup, then CUDA, then CPU
    providers = []
    if 'DirectMLExecutionProvider' in available_providers:
        providers.append('DirectMLExecutionProvider')
    if 'CUDAExecutionProvider' in available_providers:
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')
    
    print(f"Using Providers: {providers}")
    print("=========================================")

    # 1. Load Model
    print(f"Loading Model: {model_path}")
    try:
        session = ort.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        print(f"Model Loaded. Input: {input_name}, Output: {output_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Open Camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return

    print("Starting Main Loop. Press 'q' to quit.")
    
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Resize
        h, w = frame.shape[:2]
        if width is not None and w != width:
            scale = width / w
            new_h = int(h * scale)
            frame = cv2.resize(frame, (width, new_h))
        
        # Preprocess
        # BGR -> RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_rgb = img_rgb.astype(np.float32)
        
        # Expand dims: (1, H, W, 3)
        input_tensor = np.expand_dims(img_rgb, axis=0)

        # Inference
        # Run ONNX (Inputs must be numpy)
        output_tensor = session.run([output_name], {input_name: input_tensor})[0]

        # Postprocess
        output_tensor = np.clip(output_tensor, 0.0, 255.0)
        output_img = output_tensor[0].astype(np.uint8)
        
        # RGB -> BGR
        start_display = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)

        # FPS Stats
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(start_display, f"FPS: {fps:.1f} (ONNX)", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow(window_name, start_display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to .onnx style model")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Inference width.")
    
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        print(f"Warning: Model path {args.model_path} does not exist!")
        
    run_local_demo_onnx(args.model_path, args.camera, args.width)
