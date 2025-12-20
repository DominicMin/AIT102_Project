
import argparse
import os
import sys
import tensorflow as tf
import tf2onnx
import onnx

# Local Imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.style_transfer.model import make_style_transfer_network

def export_to_onnx(model_path, output_path):
    print(f"Loading Keras model from {model_path}...")
    
    # 1. Rebuild Model with explicit Input Spec for ONNX
    # While we want dynamic shape (None, None, 3), some ONNX runtimes prefer dynamic axes.
    # tf2onnx handles this.
    model = make_style_transfer_network(input_shape=(None, None, 3))
    
    # Init variables
    model(tf.zeros((1, 256, 256, 3)))
    
    # Load weights
    model.load_weights(model_path)
    
    # 2. Convert
    print("Converting to ONNX...")
    
    # Define input signature: (Batch, Height, Width, Channels)
    # Use -1 or None for dynamic dims
    input_signature = [tf.TensorSpec([1, None, None, 3], tf.float32, name="input_image")]
    
    # tf2onnx.convert.from_keras
    # opset=13 is widely supported
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)
    
    # 3. Save
    print(f"Saving to {output_path}...")
    onnx.save_model(onnx_model, output_path)
    
    return onnx_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to .h5 model")
    parser.add_argument("--output", help="Output .onnx path", default=None)
    parser.add_argument("--fp16", action="store_true", help="Optimize for Tensor Cores (RTX 3090/4060)")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.model_path.replace(".h5", ".onnx")
        
    # Phase 1: Convert H5 -> ONNX (FP32)
    model_proto = export_to_onnx(args.model_path, args.output)
    
    # Phase 2: Optimize to FP16 (Optional but recommended)
    if args.fp16:
        print("\n[Optimization] Converting to FP16 for RTX Tensor Cores...")
        try:
            from onnxconverter_common import float16
            
            # Convert
            # keep_io_types=True: Input/Output remains Float32 (Fast CPU prep), 
            # while internal layers run in Float16 (Fast GPU compute).
            model_fp16 = float16.convert_float_to_float16(model_proto, keep_io_types=True)
            
            # Save as _fp16.onnx
            base, ext = os.path.splitext(args.output)
            fp16_output = f"{base}_fp16{ext}"
            
            onnx.save(model_fp16, fp16_output)
            print(f"[Success] Optimized model saved to: {fp16_output}")
            print("Use this file for maximum FPS on RTX 4060!")
            
        except ImportError:
             print("[Error] onnxconverter-common not installed. Run: pip install onnxconverter-common")
        except Exception as e:
            print(f"[Error] FP16 conversion failed: {e}")
    else:
        print("\n[Tip] Run with --fp16 to boost performance on RTX GPUs.")
