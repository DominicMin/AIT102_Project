
import tensorflow as tf
import os
import argparse
import sys
import numpy as np
from PIL import Image

# Ensure we can import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.style_transfer.model import make_style_transfer_network
from src.style_transfer.utils import load_img, tensor_to_image

def run_inference(model_path, content_path, output_path):
    print(f"Initializing Model...")
    # Initialize the model architecture with dynamic input shape
    # This allows processing images of any resolution (Fully Convolutional)
    transformer = make_style_transfer_network(input_shape=(None, None, 3))
    
    # Build with a dummy input to initialize variables (using arbitrary size)
    transformer(tf.zeros((1, 256, 256, 3)))
    
    print(f"Loading Weights: {model_path}")
    try:
        transformer.load_weights(model_path)
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    print(f"Processing Content Image: {content_path}")
    # load_img returns float32 [0, 1] tensor of shape (1, H, W, 3)
    content_tensor = load_img(content_path) 
    
    # IMPORTANT: Training was done on [0, 255] range!
    # train_v2.py: style_img = load_img(...) * 255.0
    # And compute_loss uses vgg16.preprocess_input which expects 0-255 (if mode='caffe')
    # So we must scale input here too.
    content_tensor = content_tensor * 255.0
    
    print("Running Inference...")
    generated_tensor = transformer(content_tensor, training=False)
    
    # Post-process
    # The output of transformer is likely not clipped or cast yet depending on the final layer.
    # Usually it outputs raw float values.
    # tensor_to_image utils function usually expects [0, 1] if input is float??
    # Let's check utils.py: tensor_to_image says "tensor = tensor * 255". 
    # If our output is already ~0-255 range (because input was), then multiplying by 255 again would be wrong.
    # Wait, the transformer network usually outputs pixels in input range if it's an image-to-image net.
    # If input is 0-255, output is 0-255.
    # `tensor_to_image` implementation:
    #   tensor = tensor * 255
    #   tensor = np.array(tensor, dtype=np.uint8)
    # This implies `tensor_to_image` expects [0,1] float input.
    # So if our model outputs [0, 255], we should divide by 255 before passing to `tensor_to_image` 
    # OR manual conversion.
    
    # Let's assume output is roughly 0-255.
    generated_tensor = tf.clip_by_value(generated_tensor, 0.0, 255.0)
    generated_image = Image.fromarray(np.array(generated_tensor[0], dtype=np.uint8))
    
    print(f"Saving to {output_path}")
    generated_image.save(output_path)
    print("Success.")

def main():
    parser = argparse.ArgumentParser(description="Test trained style transfer model.")
    parser.add_argument("model_path", help="Path to .h5 model file")
    parser.add_argument("content_path", help="Path to content image (jpg/png)")
    parser.add_argument("-o", "--output", default="output.png", help="Path to save output result")
    
    args = parser.parse_args()
    
    run_inference(args.model_path, args.content_path, args.output)

if __name__ == "__main__":
    main()
