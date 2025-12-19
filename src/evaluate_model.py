
import tensorflow as tf
import os
from style_transfer.model import make_style_transfer_network
from style_transfer.utils import load_img, tensor_to_image
import matplotlib.pyplot as plt

def main():
    # Paths
    # Ensure you have downloaded the 'checkpoints' folder from the server to src/checkpoints
    checkpoint_dir = 'checkpoints'
    content_path = 'content.jpg' # content image to style
    output_path = 'custom_stylized.png'

    # Manual Checkpoint Override (Optional)
    # Set this to a specific path like 'checkpoints/ckpt-40' to load that specific one.
    # Set to None to use the latest from the manager.
    # After fine-tuning, the new model will be in 'checkpoints_finetuned/ckpt-1'
    checkpoint_dir = 'checkpoints_v2_server' 
    manual_checkpoint_path = 'checkpoints_v2_server/ckpt-40' 

    # 1. Build the Model (Must match training architecture)
    # We use (None, None, 3) to allow inference on images of any size
    print("Building model...")
    transformer = make_style_transfer_network(input_shape=(None, None, 3))

    # 2. Restore Checkpoint
    print(f"Loading checkpoint from {checkpoint_dir}...")
    ckpt = tf.train.Checkpoint(transformer=transformer)
    
    if manual_checkpoint_path:
        checkpoint_to_load = manual_checkpoint_path
        print(f"Attempting to load manual checkpoint: {checkpoint_to_load}")
    else:
        manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)
        checkpoint_to_load = manager.latest_checkpoint
        print(f"Attempting to load latest checkpoint: {checkpoint_to_load}")

    if checkpoint_to_load:
        try:
            # .expect_partial() silences warnings about optimizer variables not being loaded
            ckpt.restore(checkpoint_to_load).expect_partial()
            print(f"Successfully restored from {checkpoint_to_load}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return
    else:
        print("Error: No checkpoints found. Check your directory.")
        return

    # 3. Process Image
    print("Processing image...")
    content_image = load_img(content_path) # Returns 0-1 float32 (1, H, W, 3)
    # Model expects 0-255 input? 
    # Let's check train.py -> train_step -> calls transformer(content_images)
    # In load_dataset: img = img * 255.0. So model expects 0-255.
    content_image = content_image * 255.0

    # Inference
    stylized_tensor = transformer(content_image)

    # 4. Save Result
    # Model output is 0-255 (tanh + 1 * 127.5) -> tensor_to_image expects 0-1?
    # utils.py tensor_to_image: "tensor = tensor * 255" NO, wait.
    # Let's check utils.py again.
    # tensor_to_image(tensor): tensor = np.array(tensor*255, dtype=np.uint8) if input is 0-1?
    # Actually wait.
    
    # Let's double check utils.py
    # def tensor_to_image(tensor):
    #   tensor = tensor * 255
    #   tensor = np.array(tensor, dtype=np.uint8)
    
    # If model output is ALREADY 0-255, then utils.py will multiply by 255 AGAIN, causing white burnout.
    # We need to be careful here.
    
    # In clean-up:
    # Model output is indeed 0-255 per model.py: outputs = (x + 1.0) * 127.5
    # Utils.py tensor_to_image assumes 0-1 input typically.
    
    # Let's fix this in this script manually to be safe.
    result_image = tf.cast(stylized_tensor, tf.uint8)
    if len(result_image.shape) > 3:
        result_image = result_image[0]
    
    import PIL.Image
    img_pil = PIL.Image.fromarray(result_image.numpy())
    img_pil.save(output_path)
    
    print(f"Done! Saved stylized image to {output_path}")

if __name__ == '__main__':
    main()
