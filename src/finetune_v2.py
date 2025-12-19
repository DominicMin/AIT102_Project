
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Model
from style_transfer.train import StyleTransferTrainer
from style_transfer.utils import load_img
import os

# --- V2 Configuration based on docs/overview.md ---
# Reference uses VGG16
# Content Layer: relu2_2 (block2_conv2) -> Preserves structure better than block4
# Style Layers: relu1_2, relu2_2, relu3_3, relu4_3
CONTENT_LAYERS_V2 = ['block2_conv2'] 
STYLE_LAYERS_V2 = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']

# Weights from Reference
# Note: Reference divides Gram by C*H*W. Our original train.py divided by H*W.
# We will adjust our Gram calculation in this script to specific reference behavior.
STYLE_WEIGHT_V2 = 1e6  
CONTENT_WEIGHT_V2 = 1e0 
PIXEL_WEIGHT_V2 = 1e0  # New: Pixel Loss
TOTAL_VARIATION_WEIGHT_V2 = 2e-2

def get_vgg16_loss_model():
    """VGG16 model matching reference layers."""
    vgg = VGG16(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    # Debug: print layer names if needed
    # for layer in vgg.layers: print(layer.name)
    
    outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS_V2 + CONTENT_LAYERS_V2]
    model = Model([vgg.input], outputs)
    return model

def gram_matrix_v2(input_tensor):
    """Gram matrix normalized by C*H*W (Reference style)."""
    # input_tensor: (Batch, H, W, Channels)
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
        
    def compute_loss(self, generated_images, content_images, style_targets):
        # VGG Inputs
        vgg_gen = tf.keras.applications.vgg16.preprocess_input(generated_images)
        vgg_content = tf.keras.applications.vgg16.preprocess_input(content_images)

        gen_outputs = self.loss_model(vgg_gen)
        content_outputs = self.loss_model(vgg_content)

        gen_style_outputs = gen_outputs[:len(STYLE_LAYERS_V2)]
        gen_content_outputs = gen_outputs[len(STYLE_LAYERS_V2):]
        true_content_outputs = content_outputs[len(STYLE_LAYERS_V2):]

        # 1. Content Loss (MSE)
        content_loss = tf.add_n([tf.reduce_mean((gen_content_outputs[i] - true_content_outputs[i])**2) 
                                 for i in range(len(CONTENT_LAYERS_V2))])

        # 2. Style Loss (MSE of Grams)
        style_loss = tf.add_n([tf.reduce_mean((gram_matrix_v2(gen_style_outputs[i]) - style_targets[i])**2)
                               for i in range(len(STYLE_LAYERS_V2))])

        # 3. Pixel Loss (New)
        # MSE between generated image and original content image
        # Scale: generated_images are 0-255? Yes.
        pixel_loss = tf.reduce_mean((generated_images - content_images)**2)

        # 4. TV Loss
        tv_loss = tf.reduce_mean(tf.image.total_variation(generated_images))

        total_loss = (CONTENT_WEIGHT_V2 * content_loss) + \
                     (STYLE_WEIGHT_V2 * style_loss) + \
                     (PIXEL_WEIGHT_V2 * pixel_loss) + \
                     (TOTAL_VARIATION_WEIGHT_V2 * tv_loss)
        
        return total_loss, content_loss, style_loss

    def train(self):
        # Re-implement train setup to use VGG16 style targets
        print("Loading style image (V2)...")
        style_img = load_img(self.style_image_path) * 255.0
        
        vgg_style = tf.keras.applications.vgg16.preprocess_input(style_img)
        style_outputs = self.loss_model(vgg_style)
        # Calculate targets using V2 Gram (normalized by C*H*W)
        style_targets = [gram_matrix_v2(style_outputs[i]) for i in range(len(STYLE_LAYERS_V2))]
        
        # Dataset
        dataset = self.load_dataset()
        
        print(f"Starting V2 Fine-tuning...")
        for epoch in range(self.epochs):
            for step, content_batch in enumerate(dataset):
                loss, c_loss, s_loss = self.train_step(content_batch, style_targets)
                if step % 10 == 0:
                    print(f"Step {step}: Total Loss {loss.numpy():.1f} (Content {c_loss.numpy():.1f}, Style {s_loss.numpy():.1f})")
            
            self.ckpt_manager.save()
            print(f"Saved Checkpoint for Epoch {epoch+1}")

def main():
    checkpoint_dir = 'checkpoints'      # Old weights
    fine_tune_dir = 'checkpoints_v2'    # New weights
    
    # Initialize V2 Trainer
    trainer = StyleTransferTrainerV2(
        style_image_path='style.jpg',
        content_dir=os.path.join('data', 'val2017'),
        epochs=1,
        batch_size=32,
        check_point_dir=fine_tune_dir
    )
    
    # Load Weights
    # We load weights into the TRANSFORMER only. 
    # The Loss Model (VGG) is new (VGG16) and initialized with ImageNet weights automatically.
    # The Optimizer state might be incompatible due to loss scale changes, so we might want to reset optimizer.
    # But tf.train.Checkpoint(optimizer=...) will try to load it.
    # Let's try partial load and ignore optimizer? 
    
    specific_checkpoint = os.path.join(checkpoint_dir, 'ckpt-39')
    if os.path.exists(specific_checkpoint + '.index'):
        print(f"Loading transformer weights from {specific_checkpoint}...")
        # Only restore the transformer, ignore optimizer (start fresh for V2)
        # Create a temp checkpoint just for transformer
        temp_ckpt = tf.train.Checkpoint(transformer=trainer.transformer)
        temp_ckpt.restore(specific_checkpoint).expect_partial()
    else:
        print(f"Error: {specific_checkpoint} not found.")
        return

    trainer.train()

if __name__ == '__main__':
    main()
