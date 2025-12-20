
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
    print("Export Success!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to .h5 model")
    parser.add_argument("--output", help="Output .onnx path", default=None)
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.model_path.replace(".h5", ".onnx")
        
    export_to_onnx(args.model_path, args.output)
