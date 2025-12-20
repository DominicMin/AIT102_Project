
import onnx
from onnxconverter_common import float16
import argparse
import os

def optimize_to_fp16(input_path):
    print(f"Loading model: {input_path}")
    model = onnx.load(input_path)
    
    # Convert to FP16
    print("Converting to Float16 (FP16)...")
    model_fp16 = float16.convert_float_to_float16(model)
    
    # Save
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_fp16{ext}"
    onnx.save(model_fp16, output_path)
    
    print(f"Success! Model saved to: {output_path}")
    print("Use this model for 2x-3x faster inference on RTX GPUs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX to FP16 for RTX Performance")
    parser.add_argument("model_path", help="Path to input .onnx model")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: File not found {args.model_path}")
    else:
        optimize_to_fp16(args.model_path)
