
import os
import tensorflow as tf
from style_transfer.train import StyleTransferTrainer

def main():
    # Paths (relative to src/)
    # Ensure you run this script from the 'src' directory
    style_image_path = 'style.jpg' 
    content_dir = os.path.join('data', 'val2017')
    checkpoint_dir = 'checkpoints'

    # Check if paths exist
    if not os.path.exists(style_image_path):
        print(f"Error: Style image not found at {style_image_path}")
        return
    if not os.path.exists(content_dir):
        print(f"Error: Content directory not found at {content_dir}")
        return

    print(f"Style Image: {style_image_path}")
    print(f"Content Dir: {content_dir}")
    print("GPU Available: ", tf.config.list_physical_devices('GPU'))

    # Initialize Trainer
    # Epochs=2 is enough for a good PoC on ~5000 images
    trainer = StyleTransferTrainer(
        style_image_path=style_image_path,
        content_dir=content_dir,
        epochs=2,
        batch_size=4, # RTX 4060 has 8GB VRAM, batch size 4 is safe for 256x256
        check_point_dir=checkpoint_dir
    )

    # Start Training
    trainer.train()

if __name__ == '__main__':
    main()
