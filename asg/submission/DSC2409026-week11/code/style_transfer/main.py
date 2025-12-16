import functools
import os
import time
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from .utils import load_img, tensor_to_image, imshow

def main():
    print("TF Version: ", tf.__version__)
    print("TF Hub Version: ", hub.__version__)
    print("Eager mode enabled: ", tf.executing_eagerly())
    print("GPU available: ", tf.config.list_physical_devices('GPU'))

    # Hardcoded paths for MVP (can be changed to argparse later)
    # You need to place 'content.jpg' and 'style.jpg' in the src directory or update these paths
    content_path = 'content.jpg'
    style_path = 'style.jpg'

    # Check if files exist
    if not os.path.exists(content_path):
        print(f"Error: Content image '{content_path}' not found. Please add a 'content.jpg' file.")
        return
    if not os.path.exists(style_path):
        print(f"Error: Style image '{style_path}' not found. Please add a 'style.jpg' file.")
        return

    content_image = load_img(content_path)
    style_image = load_img(style_path)

    print("Loading Hub Model...")
    # Load the fast style transfer model from TF-Hub
    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    print("Model loaded.")

    print("Stylizing image...")
    start_time = time.time()
    
    # The signature of the model is styling(content_image, style_image)
    stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
    
    end_time = time.time()
    print(f"Stylization done in {end_time - start_time:.4f} seconds")

    # Save and show result
    output_filename = 'stylized_image.png'
    result_image = tensor_to_image(stylized_image)
    result_image.save(output_filename)
    print(f"Saved result to {output_filename}")

    # Optional: Display using matplotlib if running in an environment that supports it
    # plt.figure(figsize=(10, 10))
    # plt.subplot(1, 3, 1)
    # imshow(content_image, 'Content Image')
    # plt.subplot(1, 3, 2)
    # imshow(style_image, 'Style Image')
    # plt.subplot(1, 3, 3)
    # imshow(stylized_image, 'Stylized Image')
    # plt.show()

if __name__ == '__main__':
    main()
