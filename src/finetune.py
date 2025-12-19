
import tensorflow as tf
from style_transfer.train import StyleTransferTrainer, STYLE_LAYERS
import os

def fine_tune():
    # Configuration
    style_image_path = 'style.jpg' 
    content_dir = os.path.join('data', 'val2017')
    checkpoint_dir = 'checkpoints'      # Where the bad model is
    fine_tune_dir = 'checkpoints_finetuned' # Where to save fixed model

    # 1. Initialize Trainer
    trainer = StyleTransferTrainer(
        style_image_path=style_image_path,
        content_dir=content_dir,
        epochs=1,           # Only 1 epoch needed to fix weights
        batch_size=4,       # Local RTX 4060 is fine for this
        check_point_dir=fine_tune_dir
    )

    # 2. Force Load Weights from the "Bad" Checkpoint
    # We want to start from where we left off, not scratch
    # 2. Force Load Weights from the "Bad" Checkpoint
    # We force load ckpt-39 as requested
    specific_checkpoint = os.path.join(checkpoint_dir, 'ckpt-39')
    
    # Check if files exist to avoid vague errors
    if os.path.exists(specific_checkpoint + '.index'):
        print(f"Loading weights from {specific_checkpoint} for fine-tuning...")
        trainer.ckpt.restore(specific_checkpoint).expect_partial()
    else:
        print(f"Error: Checkpoint {specific_checkpoint} not found!")
        print(f"Please ensure {specific_checkpoint}.index exists.")
        return

    # 3. ADJUST WEIGHTS (The Fix)
    # Original: Style=1e-2, Content=1e4
    # Problem: Style too strong (Noise).
    # Fix: Reduce Style Weight drastically.
    # Note: We need to patch the compute_loss method or variables used in it.
    # Since constants are in `train.py`, we can't easily change them dynamically unless we patch the module or parameters.
    
    # Let's monkey-patch the module variable for this session
    import style_transfer.train as train_module
    print(f"Old Style Weight: {train_module.STYLE_WEIGHT}")
    train_module.STYLE_WEIGHT = 1e-3 # Reduce by 10x (Balanced)
    train_module.TOTAL_VARIATION_WEIGHT = 100 # Moderate smoothness
    print(f"New Style Weight: {train_module.STYLE_WEIGHT}")
    print(f"New TV Weight: {train_module.TOTAL_VARIATION_WEIGHT}")

    # 4. Run Training
    print("Starting Fine-tuning...")
    trainer.train()

if __name__ == '__main__':
    fine_tune()
