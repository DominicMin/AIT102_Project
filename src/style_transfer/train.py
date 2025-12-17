import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model
import time
import os
import glob
from tqdm import tqdm
from .model import make_style_transfer_network
from .utils import load_img, tensor_to_image

import datetime

# Configuration
CONTENT_LAYERS = ['block4_conv2']
STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Weights
STYLE_WEIGHT = 1e-2
CONTENT_WEIGHT = 1e4
TOTAL_VARIATION_WEIGHT = 30

def get_vgg_loss_model():
    """Creates a VGG model that returns a list of intermediate output values."""
    vgg = VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS + CONTENT_LAYERS]
    model = Model([vgg.input], outputs)
    return model

def gram_matrix(input_tensor):
    """Calculates the Gram Matrix for style loss."""
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

class StyleTransferTrainer:
    def __init__(self, style_image_path, content_dir, epochs=2, batch_size=4, check_point_dir='checkpoints', log_dir='logs'):
        self.style_image_path = style_image_path
        self.content_dir = content_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.check_point_dir = check_point_dir
        
        # TensorBoard Logger
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = os.path.join(log_dir, current_time)
        self.summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        
        # Initialize Model
        self.transformer = make_style_transfer_network()
        self.loss_model = get_vgg_loss_model()
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.5)
        
        # Checkpoint
        self.ckpt = tf.train.Checkpoint(transformer=self.transformer, optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.check_point_dir, max_to_keep=5)

    def load_dataset(self):
        """Loads images from content_dir."""
        # Find all jpg/png files
        image_files = glob.glob(os.path.join(self.content_dir, '**', '*.jpg'), recursive=True)
        if not image_files:
            raise FileNotFoundError(f"No .jpg images found in {self.content_dir}")
            
        print(f"Found {len(image_files)} training images.")
        
        def process_path(file_path):
            img = tf.io.read_file(file_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32) # 0-1
            img = tf.image.resize(img, [256, 256]) # Resize to fixed size for batching
            img = img * 255.0 # Scale to 0-255
            return img

        ds = tf.data.Dataset.from_tensor_slices(image_files)
        ds = ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def compute_loss(self, generated_images, content_images, style_targets):
        # Preprocess for VGG (expects 0-255 BGR centered)
        # Note: generated_images and content_images are 0-255 RGB
        
        # VGG Inputs
        vgg_gen = tf.keras.applications.vgg19.preprocess_input(generated_images)
        vgg_content = tf.keras.applications.vgg19.preprocess_input(content_images) # Content targets come from original image
        
        # Forward pass through VGG
        gen_outputs = self.loss_model(vgg_gen)
        # Content pass (we only need content layers)
        content_outputs = self.loss_model(vgg_content)
        
        # Split outputs
        gen_style_outputs = gen_outputs[:len(STYLE_LAYERS)]
        gen_content_outputs = gen_outputs[len(STYLE_LAYERS):]
        
        true_content_outputs = content_outputs[len(STYLE_LAYERS):]
        
        # Content Loss
        content_loss = tf.add_n([tf.reduce_mean((gen_content_outputs[i] - true_content_outputs[i])**2) 
                                 for i in range(len(CONTENT_LAYERS))])
        
        # Style Loss
        style_loss = tf.add_n([tf.reduce_mean((gram_matrix(gen_style_outputs[i]) - style_targets[i])**2)
                               for i in range(len(STYLE_LAYERS))])
        
        # Total Variation Loss (smoothness)
        # tf.image.total_variation returns sum of abs differences
        tv_loss = tf.reduce_mean(tf.image.total_variation(generated_images))
        
        total_loss = (CONTENT_WEIGHT * content_loss) + (STYLE_WEIGHT * style_loss) + (TOTAL_VARIATION_WEIGHT * tv_loss)
        return total_loss, content_loss, style_loss

    @tf.function
    def train_step(self, content_images, style_targets):
        with tf.GradientTape() as tape:
            generated_images = self.transformer(content_images)
            loss, c_loss, s_loss = self.compute_loss(generated_images, content_images, style_targets)
            
        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))
        return loss, c_loss, s_loss

    def train(self):
        # Load Style Target
        print("Loading style image...")
        style_img = load_img(self.style_image_path) # Shape (1, H, W, 3), 0-255 (load_img returns 0-1? check utils)
        # Check utils.py: load_img returns 0-1 float32 tensor
        style_img = style_img * 255.0 # Scale to 0-255
        
        # Precompute Style Targets (Gram Matrices)
        vgg_style = tf.keras.applications.vgg19.preprocess_input(style_img)
        style_outputs = self.loss_model(vgg_style)
        style_targets = [gram_matrix(style_outputs[i]) for i in range(len(STYLE_LAYERS))]
        
        # Dataset
        dataset = self.load_dataset()
        
        print(f"Starting training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            prog_bar = tqdm(dataset)
            
            for step, content_batch in enumerate(prog_bar):
                loss, c_loss, s_loss = self.train_step(content_batch, style_targets)
                prog_bar.set_description(f"Loss: {loss.numpy():.2f} (C: {c_loss.numpy():.2f}, S: {s_loss.numpy():.2f})")
                
                # TensorBoard Logging
                with self.summary_writer.as_default():
                    tf.summary.scalar('total_loss', loss, step=epoch*len(dataset) + step)
                    tf.summary.scalar('content_loss', c_loss, step=epoch*len(dataset) + step)
                    tf.summary.scalar('style_loss', s_loss, step=epoch*len(dataset) + step)
                    
                    # Log images every 100 steps
                    if step % 100 == 0:
                        # Log the first image in batch
                        example_content = content_batch[0]
                        example_generated = self.transformer(tf.expand_dims(example_content, 0))[0]
                        
                        # Images are 0-255, convert to 0-1 for tensorboard display if float
                        # Or keep as uint8. Let's cast to uint8
                        tf.summary.image("Training/Input", tf.cast(content_batch[:1], tf.uint8), step=epoch*len(dataset) + step)
                        tf.summary.image("Training/Output", tf.cast(tf.expand_dims(example_generated, 0), tf.uint8), step=epoch*len(dataset) + step)
                
            # Save checkpoint each epoch
            ckpt_save_path = self.ckpt_manager.save()
            print(f"Saved checkpoint for epoch {epoch+1} at {ckpt_save_path}")
            
            # Optional: Save a sample result
            # Can add visualization code here
            
        print("Training complete.")

if __name__ == "__main__":
    # Example usage (will be called from separate script usually)
    pass
