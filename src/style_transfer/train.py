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
        
        # Dual GPU Strategy
        self.strategy = tf.distribute.MirroredStrategy()
        print(f"Number of devices: {self.strategy.num_replicas_in_sync}")
        
        # Adjust batch size for global context
        self.batch_size = batch_size
        self.global_batch_size = batch_size # Input is treated as 'global' or 'per_replica'?
        # Standard convention: input batch_size is usually global batch size in Keras fit,
        # but in custom loops, we often specify global and let it split.
        # Let's assume the user passes the GLOBAL batch size (e.g. 32).
        
        self.check_point_dir = check_point_dir
        
        # TensorBoard Logger
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = os.path.join(log_dir, current_time)
        self.summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        
        with self.strategy.scope():
            # Initialize Model inside strategy scope
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

        dataset = tf.data.Dataset.from_tensor_slices(image_files)
        dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
        # Batch size is GLOBAL batch size here
        dataset = dataset.shuffle(buffer_size=1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    def compute_loss(self, generated_images, content_images, style_targets):
        # Preprocess
        vgg_gen = tf.keras.applications.vgg19.preprocess_input(generated_images)
        vgg_content = tf.keras.applications.vgg19.preprocess_input(content_images) 
        
        gen_outputs = self.loss_model(vgg_gen)
        content_outputs = self.loss_model(vgg_content)
        
        gen_style_outputs = gen_outputs[:len(STYLE_LAYERS)]
        gen_content_outputs = gen_outputs[len(STYLE_LAYERS):]
        true_content_outputs = content_outputs[len(STYLE_LAYERS):]
        
        # Content Loss
        content_loss = tf.add_n([tf.reduce_mean((gen_content_outputs[i] - true_content_outputs[i])**2) 
                                 for i in range(len(CONTENT_LAYERS))])
        
        # Style Loss
        style_loss = tf.add_n([tf.reduce_mean((gram_matrix(gen_style_outputs[i]) - style_targets[i])**2)
                               for i in range(len(STYLE_LAYERS))])
        
        # TV Loss
        tv_loss = tf.reduce_mean(tf.image.total_variation(generated_images))
        
        total_loss = (CONTENT_WEIGHT * content_loss) + (STYLE_WEIGHT * style_loss) + (TOTAL_VARIATION_WEIGHT * tv_loss)
        
        # Scale loss by 1/global_batch_size is usually handled by reduce_mean if summing?
        # tf.reduce_mean computes mean per batch. 
        # In multi-gpu, standard practice is sum per replica, then divide by global batch size.
        # But here we used reduce_mean per replica. 
        # The gradients will be averaged across replicas by default.
        # So per-replica mean is fine.
        
        return total_loss, content_loss, style_loss

    def train_step(self, content_images, style_targets):
        # This function runs PER REPLICA
        with tf.GradientTape() as tape:
            generated_images = self.transformer(content_images)
            loss, c_loss, s_loss = self.compute_loss(generated_images, content_images, style_targets)
            
        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))
        return loss, c_loss, s_loss

    def train(self):
        # Load Style Target
        print("Loading style image...")
        style_img = load_img(self.style_image_path)
        style_img = style_img * 255.0 # Scale to 0-255
        
        # Precompute Style Targets (Gram Matrices) - Can be done on CPU or single GPU
        vgg_style = tf.keras.applications.vgg19.preprocess_input(style_img)
        # Using the strategy scope model just in case, though for inference it matters less
        # Actually better to compute this once and pass it in as constant tensors
        with self.strategy.scope():
             style_outputs = self.loss_model(vgg_style)
             style_targets = [gram_matrix(style_outputs[i]) for i in range(len(STYLE_LAYERS))]
             # Ensure targets are constant tensors available to all replicas
             style_targets = [tf.constant(t) for t in style_targets]
        
        # Dataset
        dataset = self.load_dataset()
        # Distribute dataset
        dist_dataset = self.strategy.experimental_distribute_dataset(dataset)
        
        print(f"Starting distributed training for {self.epochs} epochs with global batch size {self.batch_size}...")
        
        # Define distributed step
        @tf.function
        def distributed_train_step(content_batch, style_targets_arg):
            per_replica_losses = self.strategy.run(self.train_step, args=(content_batch, style_targets_arg))
            # Reduce for logging
            return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            
            # Use simple loop instead of tqdm on dist_dataset directly if possible, or manual clear
            # tqdm with dist_dataset can be tricky because length might be unknown or infinite if repeated
            
            step = 0
            for content_batch in dist_dataset:
                # Returns (total_loss, c_loss, s_loss) tuple of reduced values? 
                # Strategy.run returns a tuple of PerReplica values if train_step returns a tuple.
                # reduce needs to be called on each element.
                
                # Let's adjust distributed_train_step to return list of reduced values
                
                # Re-define inside loop or class to handle the tuple return:
                # Actually, let's just do it cleanly:
                
                per_replica_results = self.strategy.run(self.train_step, args=(content_batch, style_targets))
                # per_replica_results is a tuple of (loss, c_loss, s_loss), each is PerReplica
                
                loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[0], axis=None)
                c_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[1], axis=None)
                s_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[2], axis=None)
                
                if step % 20 == 0:
                     print(f"Step {step}: Loss: {loss:.2f} (C: {c_loss:.2f}, S: {s_loss:.2f})")

                # TensorBoard Logging
                with self.summary_writer.as_default():
                    tf.summary.scalar('total_loss', loss, step=epoch*1000 + step) # approx step
                    tf.summary.scalar('content_loss', c_loss, step=epoch*1000 + step)
                    tf.summary.scalar('style_loss', s_loss, step=epoch*1000 + step)
                    
                    if step % 100 == 0:
                        # Log images (Just take one replica's input)
                        # content_batch is PerReplica.
                        # We can grab local values.
                        if hasattr(content_batch, 'values'):
                            local_content = content_batch.values[0]
                        else:
                            local_content = content_batch
                            
                        example_content = local_content[0]
                        # Run inference on one image
                        # Must be inside scope or just run transformer directly if weights are synced (they are)
                        example_generated = self.transformer(tf.expand_dims(example_content, 0))[0]
                        
                        tf.summary.image("Training/Input", tf.cast(tf.expand_dims(example_content, 0), tf.uint8), step=epoch*1000 + step)
                        tf.summary.image("Training/Output", tf.cast(tf.expand_dims(example_generated, 0), tf.uint8), step=epoch*1000 + step)

                step += 1
                
            # Save checkpoint each epoch
            ckpt_save_path = self.ckpt_manager.save()
            print(f"Saved checkpoint for epoch {epoch+1} at {ckpt_save_path}")
            
        print("Training complete.")

if __name__ == "__main__":
    pass
