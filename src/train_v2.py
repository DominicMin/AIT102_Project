
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Model
from style_transfer.train import StyleTransferTrainer
from style_transfer.utils import load_img
import os
import datetime

# --- V2 Configuration based on docs/overview.md ---
# Reference uses VGG16
CONTENT_LAYERS_V2 = ['block2_conv2'] 
STYLE_LAYERS_V2 = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']

# Weights from Reference (Corrected based on TensorBoard Data)
# Raw Content Loss ~ 4e5
# Raw Style Loss ~ 1e5
# To balance them (~1:1 ratio), Style Weight should be around 4.0 - 5.0
# Previous 4e4 was 10,000x too high!
STYLE_WEIGHT_V2 = 5.0   
CONTENT_WEIGHT_V2 = 1.0 
PIXEL_WEIGHT_V2 = 100.0   # Re-enable Pixel Loss to lock in structure (Magnitude ~2000 -> *100 = 2e5)
TOTAL_VARIATION_WEIGHT_V2 = 1e-3

def get_vgg16_loss_model():
    """VGG16 model matching reference layers."""
    vgg = VGG16(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS_V2 + CONTENT_LAYERS_V2]
    model = Model([vgg.input], outputs)
    return model

def gram_matrix_v2(input_tensor):
    """Gram matrix normalized by C*H*W (Reference style)."""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    H, W, C = input_shape[1], input_shape[2], input_shape[3]
    num_elements = tf.cast(H * W * C, tf.float32)
    return result / num_elements

class StyleTransferTrainerV2(StyleTransferTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Override loss model with VGG16
        self.loss_model = get_vgg16_loss_model()
        
        # Optimizer
        # Lower LR for stability (1e-3 might be too aggressive for VGG16 scratch)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
        
    def compute_loss(self, generated_images, content_images, style_targets):
        # VGG Inputs
        vgg_gen = tf.keras.applications.vgg16.preprocess_input(generated_images)
        vgg_content = tf.keras.applications.vgg16.preprocess_input(content_images)

        gen_outputs = self.loss_model(vgg_gen)
        content_outputs = self.loss_model(vgg_content)

        gen_style_outputs = gen_outputs[:len(STYLE_LAYERS_V2)]
        gen_content_outputs = gen_outputs[len(STYLE_LAYERS_V2):]
        true_content_outputs = content_outputs[len(STYLE_LAYERS_V2):]

        # 1. Content Loss
        content_loss = tf.add_n([tf.reduce_mean((gen_content_outputs[i] - true_content_outputs[i])**2) 
                                 for i in range(len(CONTENT_LAYERS_V2))])

        # 2. Style Loss
        style_loss = tf.add_n([tf.reduce_mean((gram_matrix_v2(gen_style_outputs[i]) - style_targets[i])**2)
                               for i in range(len(STYLE_LAYERS_V2))])

        # 3. Pixel Loss
        pixel_loss = tf.reduce_mean((generated_images - content_images)**2)

        # 4. TV Loss
        tv_loss = tf.reduce_mean(tf.image.total_variation(generated_images))

        total_loss = (CONTENT_WEIGHT_V2 * content_loss) + \
                     (STYLE_WEIGHT_V2 * style_loss) + \
                     (PIXEL_WEIGHT_V2 * pixel_loss) + \
                     (TOTAL_VARIATION_WEIGHT_V2 * tv_loss)
        
        return total_loss, content_loss, style_loss

    def train(self):
        # Re-implement train setup/loop to use VGG16 style targets
        print("Loading style image (V2)...")
        style_img = load_img(self.style_image_path) * 255.0
        
        vgg_style = tf.keras.applications.vgg16.preprocess_input(style_img)
        style_outputs = self.loss_model(vgg_style)
        # Calculate targets using V2 Gram
        style_targets = [gram_matrix_v2(style_outputs[i]) for i in range(len(STYLE_LAYERS_V2))]
        
        dataset = self.load_dataset()
        
        print(f"Starting V2 Training (Scratch) for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            for step, content_batch in enumerate(dataset):
                loss, c_loss, s_loss = self.train_step(content_batch, style_targets)
                
                # TensorBoard
                with self.summary_writer.as_default():
                    current_step = epoch * 1000 + step # Alignment
                    tf.summary.scalar('total_loss', loss, step=current_step) 
                    tf.summary.scalar('content_loss', c_loss, step=current_step)
                    tf.summary.scalar('style_loss', s_loss, step=current_step)
                    
                    # Log images every 50 steps (More frequent for debugging)
                    if step % 50 == 0:
                        example_content = content_batch[0]
                        # Run inference
                        example_generated = self.transformer(tf.expand_dims(example_content, 0))[0]
                        
                        # Cast to uint8 for display (0-255 range)
                        tf.summary.image("Training/Input", tf.cast(tf.expand_dims(example_content, 0), tf.uint8), step=current_step)
                        tf.summary.image("Training/Output", tf.cast(tf.expand_dims(example_generated, 0), tf.uint8), step=current_step)

                if step % 20 == 0:
                    print(f"Step {step}: Total {loss:.1f} (C {c_loss:.1f}, S {s_loss:.1f})")
            
            # Save checkpoint
            self.ckpt_manager.save()
            print(f"Saved Checkpoint for Epoch {epoch+1}")

def main():
    checkpoint_dir = 'checkpoints_v2_server'    # Separate dir
    content_dir = 'data/val2017' # Ensure this matches server structure
    
    # Initialize V2 Trainer for Server
    trainer = StyleTransferTrainerV2(
        style_image_path='style.jpg',
        content_dir=content_dir,
        epochs=40,          # Full training
        batch_size=16,      # V100 32GB
        check_point_dir=checkpoint_dir,
        log_dir='logs_v2'
    )
    
    print("Starting Training from Scratch (VGG16 + Pixel Loss)...")
    trainer.train()

if __name__ == '__main__':
    main()
