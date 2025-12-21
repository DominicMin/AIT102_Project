
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG19, VGG16
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import cv2
import time
import argparse
import os
import sys
import glob
import datetime
from tqdm import tqdm

# ==========================================
# From style_transfer/model.py
# ==========================================

class InstanceNormalization(layers.Layer):
    """Instance Normalization Layer (https://arxiv.org/abs/1607.08022)"""
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer='ones',
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
        super(InstanceNormalization, self).build(input_shape)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

class ReflectionPadding2D(layers.Layer):
    """Reflection Padding as used in style transfer to reduce artifacts."""
    def __init__(self, padding=(1, 1), **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.padding = tuple(padding)

    def call(self, input_tensor):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor, [[0,0], [padding_height, padding_height], [padding_width, padding_width], [0,0]], 'REFLECT')

def conv_block(x, filters, kernel_size, strides=1, padding='same', activation=True):
    """Standard Convolution Block"""
    if padding == 'reflect':
        pad_amount = (kernel_size - 1) // 2
        x = ReflectionPadding2D(padding=(pad_amount, pad_amount))(x)
        padding_mode = 'valid'
    else:
        padding_mode = padding
        
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding_mode, use_bias=False)(x)
    x = InstanceNormalization()(x)
    if activation:
        x = layers.ReLU()(x)
    return x

def residual_block(x, filters, kernel_size=3):
    """Residual Block with Reflection Padding"""
    shortcut = x
    
    # Exact implementation of Johnson et al. Residual Block
    # 3x3 Conv -> BN (InstanceNorm) -> ReLU -> 3x3 Conv -> BN (InstanceNorm) -> Sum
    pad_amount = (kernel_size - 1) // 2
    
    y = ReflectionPadding2D(padding=(pad_amount, pad_amount))(x)
    y = layers.Conv2D(filters, kernel_size, strides=1, padding='valid', use_bias=False)(y)
    y = InstanceNormalization()(y)
    y = layers.ReLU()(y)
    
    y = ReflectionPadding2D(padding=(pad_amount, pad_amount))(y)
    y = layers.Conv2D(filters, kernel_size, strides=1, padding='valid', use_bias=False)(y)
    y = InstanceNormalization()(y)
    
    return layers.Add()([y, shortcut])

def upsample_block(x, filters, kernel_size=3):
    """Upsampling Block"""
    # Using Sub-pixel Conv (Conv2DTranspose) or Resize+Conv
    # Johnson used ConvTranspose. Let's use Resize+Conv for better quality (checkerboard artifact removal).
    x = layers.UpSampling2D(size=2)(x)
    x = conv_block(x, filters, kernel_size, strides=1, padding='reflect')
    return x

def make_style_transfer_network(input_shape=(256, 256, 3)):
    """Builds the Image Transformation Network."""
    inputs = layers.Input(shape=input_shape)
    
    # 1. Reflection Pad + Initial Conv (32 filters, 9x9)
    x = ReflectionPadding2D(padding=(4, 4))(inputs)
    x = layers.Conv2D(32, 9, strides=1, padding='valid', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)
    
    # 2. Downsampling (64 filters, 3x3, stride 2)
    x = conv_block(x, 64, 3, strides=2, padding='same') # padding same effectively handles stride 2
    
    # 3. Downsampling (128 filters, 3x3, stride 2)
    x = conv_block(x, 128, 3, strides=2, padding='same')
    
    # 4. Residual Blocks (5 blocks)
    for _ in range(5):
        x = residual_block(x, 128)
        
    # 5. Upsampling (64 filters)
    x = upsample_block(x, 64)
    
    # 6. Upsampling (32 filters)
    x = upsample_block(x, 32)
    
    # 7. Output Layer (3 filters, 9x9) -> Tanh -> Scale
    x = ReflectionPadding2D(padding=(4, 4))(x)
    x = layers.Conv2D(3, 9, strides=1, padding='valid', activation='tanh')(x)
    
    # Output is [-1, 1], scale to [0, 255] for image
    outputs = (x + 1.0) * 127.5
    
    return Model(inputs, outputs, name='StyleTransferNetwork')


# ==========================================
# From style_transfer/utils.py
# ==========================================

def tensor_to_image(tensor):
  """Converts a tensor to a PIL image."""
  tensor = tensor * 255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor) > 3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
  """Loads an image from a file, resizes it to a max dimension of 512, and normalizes it."""
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def imshow(image, title=None):
  """Displays an image suitable for matplotlib."""
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)


# ==========================================
# From style_transfer/train.py
# ==========================================

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


# ==========================================
# From train_v2.py
# ==========================================

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
CONTENT_WEIGHT_V2 = 2.0 
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
    def __init__(self, style_name="unknown", output_model_dir="models/exported", patience=10, min_delta=100.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.style_name = style_name
        self.output_model_dir = output_model_dir
        self.patience = patience
        self.min_delta = min_delta
        
        # Ensure export dir exists
        os.makedirs(self.output_model_dir, exist_ok=True)
        
        with self.strategy.scope():
            self.loss_model = get_vgg16_loss_model()
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
            self.ckpt = tf.train.Checkpoint(transformer=self.transformer, optimizer=self.optimizer)
            self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.check_point_dir, max_to_keep=5)
    
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
        print(f"Loading style image (V2): {self.style_image_path}")
        style_img = load_img(self.style_image_path) * 255.0
        
        with self.strategy.scope():
            vgg_style = tf.keras.applications.vgg16.preprocess_input(style_img)
            style_outputs = self.loss_model(vgg_style)
            style_targets = [gram_matrix_v2(style_outputs[i]) for i in range(len(STYLE_LAYERS_V2))]
            style_targets = [tf.constant(t) for t in style_targets]
        
        dataset = self.load_dataset()
        dist_dataset = self.strategy.experimental_distribute_dataset(dataset)
        
        print(f"Starting V2 Training for {self.epochs} epochs. Patience={self.patience}")
        
        @tf.function
        def distributed_train_step(content_batch, style_targets_arg):
            per_replica_results = self.strategy.run(self.train_step, args=(content_batch, style_targets_arg))
            loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[0], axis=None)
            c_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[1], axis=None)
            s_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_results[2], axis=None)
            return loss, c_loss, s_loss

        # Early Stopping Variables
        best_loss = float('inf')
        wait = 0
        final_loss = 0.0

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            step = 0
            epoch_loss_sum = 0.0
            num_steps = 0
            
            steps_per_epoch = len(dataset)
            prog_bar = tqdm(dist_dataset, total=steps_per_epoch, desc=f"Epoch {epoch+1}")
            
            for content_batch in prog_bar:
                loss, c_loss, s_loss = distributed_train_step(content_batch, style_targets)
                
                # Update metrics
                current_loss = float(loss)
                epoch_loss_sum += current_loss
                num_steps += 1
                
                prog_bar.set_postfix({"Loss": f"{current_loss:.1f}", "C": f"{float(c_loss):.1f}", "S": f"{float(s_loss):.1f}"})
                
                with self.summary_writer.as_default():
                    current_step = epoch * steps_per_epoch + step
                    tf.summary.scalar('total_loss', loss, step=current_step)
                    tf.summary.scalar('content_loss', c_loss, step=current_step)
                    tf.summary.scalar('style_loss', s_loss, step=current_step) # Fixed: Now logging all losses
                    
                    # Log images every 50 steps
                    if step % 50 == 0:
                        # Extract single image for visualization
                        # dist_dataset batches might be PerReplica objects
                        if hasattr(content_batch, 'values'):
                             local_content = content_batch.values[0] # Get first replica
                        else:
                             local_content = content_batch
                             
                        example_content = local_content[0]
                        
                        # Run inference (This runs eagerly for logging purposes)
                        # Expand dims to (1, H, W, 3)
                        input_tensor = tf.expand_dims(example_content, 0)
                        generated_image = self.transformer(input_tensor, training=False)[0]
                        
                        # Cast to uint8 for display
                        # VGG might have preprocessed it, but our dataset loader outputs [0, 255] or [0, 1]?
                        # Assuming [0, 255] based on utils.
                        
                        tf.summary.image("Training/Input", tf.cast(input_tensor, tf.uint8), step=current_step)
                        tf.summary.image("Training/Output", tf.cast(tf.expand_dims(generated_image, 0), tf.uint8), step=current_step)
                
                step += 1
            
            # Epoch End Logic
            avg_epoch_loss = epoch_loss_sum / num_steps if num_steps > 0 else float('inf')
            print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.1f}")
            final_loss = avg_epoch_loss

            # Checkpoint
            self.ckpt_manager.save()
            
            # Early Stopping Check
            if avg_epoch_loss < best_loss - self.min_delta:
                best_loss = avg_epoch_loss
                wait = 0
            else:
                wait += 1
                print(f"Early Stopping counter: {wait}/{self.patience}")
                if wait >= self.patience:
                    print(f"Stopping early! Loss hasn't improved for {self.patience} epochs.")
                    break
        
        # Export Model
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        file_name = f"{self.style_name}_{current_date}_loss{int(final_loss)}.h5"
        export_path = os.path.join(self.output_model_dir, file_name)
        
        print(f"Exporting model to {export_path}...")
        self.transformer.save_weights(export_path)
        print("Done.")

        # --- AUTO-ONNX EXPORT ---
        try:
            import tf2onnx
            
            onnx_filename = f"{self.style_name}_{current_date}_loss{int(final_loss)}.onnx"
            onnx_path = os.path.join(self.output_model_dir, onnx_filename)
            
            print(f"[Auto-Export] Converting to ONNX: {onnx_path}...")
            
            # Define input signature [1, H, W, 3] or [None, None, None, 3] for dynamic
            spec = (tf.TensorSpec((1, 256, 256, 3), tf.float32, name="input_image"),)
            
            # Convert
            import onnx
            model_proto, _ = tf2onnx.convert.from_keras(self.transformer, input_signature=spec, opset=13)
            onnx.save(model_proto, onnx_path)
            
            print(f"[Auto-Export] ONNX Saved! Size: {os.path.getsize(onnx_path)/1024/1024:.2f} MB")
            
        except ImportError:
            print("[Warning] tf2onnx not installed. Skipping auto-export.")
        except Exception as e:
            print(f"[Error] ONNX export failed: {e}")


# ==========================================
# From video_demo.py
# ==========================================

def process_video(model_path, input_video, output_video, width=None):
    print("Initializing...")
    
    # 1. Load Model
    # Dynamic shape input
    transformer = make_style_transfer_network(input_shape=(None, None, 3))
    # Build dummy
    transformer(tf.zeros((1, 256, 256, 3)))
    
    print(f"Loading weights from {model_path}...")
    transformer.load_weights(model_path)
    
    # Compilation for speed (Crucial for Video)
    # We create a concrete function for the specific input size once we know it, 
    # OR we use a dynamic shape function. 
    # For consistent speed, if video size is constant, concrete function is best.
    
    @tf.function
    def style_transform(content_image):
        return transformer(content_image, training=False)

    # 2. Open Video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error opening video file {input_video}")
        return

    # Video properties
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Target dimensions
    if width:
        scale = width / orig_width
        new_width = width
        new_height = int(orig_height * scale)
    else:
        new_width = orig_width
        new_height = orig_height
        
    print(f"Processing Video: {orig_width}x{orig_height} -> {new_width}x{new_height} @ {fps} FPS")

    # Output Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (new_width, new_height))
    
    frame_count = 0
    total_inference_time = 0
    
    print("Starting processing loop...")
    
    # Warmup
    dummy = tf.zeros((1, new_height, new_width, 3), dtype=tf.float32)
    _ = style_transform(dummy)
    print("Warmup complete.")

    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        t0 = time.time()
        
        # Preprocess
        if width:
            frame = cv2.resize(frame, (new_width, new_height))
            
        # CV2 (BGR) -> RGB -> Tensor
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.float32)
        img_tensor = tf.expand_dims(img_tensor, 0) # (1, H, W, 3)
        
        # Inference (Input is 0-255)
        output_tensor = style_transform(img_tensor)
        
        # Postprocess
        # Clip and Cast
        output_tensor = tf.clip_by_value(output_tensor, 0.0, 255.0)
        output_img = output_tensor[0].numpy().astype(np.uint8)
        
        # RGB -> BGR
        output_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        
        inference_time = time.time() - t0
        total_inference_time += inference_time
        frame_count += 1
        
        # Write
        out.write(output_bgr)
        
        # Console Logging
        if frame_count % 10 == 0:
            print(f"Frame {frame_count}: {1.0/inference_time:.1f} FPS (Inference)")
            
    total_time = time.time() - start_time
    avg_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
    
    cap.release()
    out.release()
    
    print(f"==========================================")
    print(f"Conversion Complete!")
    print(f"Processed {frame_count} frames in {total_time:.2f}s")
    print(f"Average Inference FPS: {avg_fps:.2f}")
    print(f"Output saved to: {output_video}")
    print(f"==========================================")
