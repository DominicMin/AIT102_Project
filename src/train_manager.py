import os
import argparse
import sys

# Add current directory to path to import train_v2

from utils import StyleTransferTrainerV2

def run_training_for_style(style_base_name):
    # Paths
    project_root = os.path.dirname(os.path.abspath(__file__)) # src/
    
    style_img_path = os.path.join(project_root, "styles", f"{style_base_name}.jpg")
    if not os.path.exists(style_img_path):
        # Try .png or other extensions if needed, but assuming .jpg for now
        print(f"[Error] Style image not found: {style_img_path}")
        return

    checkpoint_dir = os.path.join(project_root, "models", "checkpoints", style_base_name)
    output_dir = os.path.join(project_root, "models", "exported")
    log_dir = os.path.join(project_root, "logs_v2", style_base_name)
    content_dir = os.path.join(project_root, "data", "val2017")

    print(f"==================================================")
    print(f"  Starting Training Manager for Style: {style_base_name}")
    print(f"  - Image: {style_img_path}")
    print(f"  - Checkpoints: {checkpoint_dir}")
    print(f"==================================================")

    # Ensure directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize Trainer
    trainer = StyleTransferTrainerV2(
        style_image_path=style_img_path,
        style_name=style_base_name,
        output_model_dir=output_dir,
        content_dir=content_dir,
        epochs=60,               # As requested
        batch_size=32,           # Dual GPU Optimized
        check_point_dir=checkpoint_dir,
        log_dir=log_dir,
        patience=10              # As requested
    )

    # Start Training
    try:
        trainer.train()
    except Exception as e:
        print(f"[Exception] Training failed for {style_base_name}: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="AIT102 Training Manager")
    parser.add_argument("--style", type=str, required=True, help="Base name of the style (e.g. 'picasso' for 'styles/picasso.jpg')")
    
    args = parser.parse_args()
    
    run_training_for_style(args.style)

if __name__ == "__main__":
    main()
