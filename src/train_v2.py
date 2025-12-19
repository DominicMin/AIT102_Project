
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Model
from style_transfer.train import StyleTransferTrainer
from style_transfer.utils import load_img
from tqdm import tqdm
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


import argparse

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style", type=str, default="styles/ukiyoe.jpg")
    parser.add_argument("--style_name", type=str, default="ukiyoe", help="Name for export (e.g. vangogh)")
    parser.add_argument("--content_dir", type=str, default="data/val2017")
    parser.add_argument("--checkpoint_dir", type=str, default="models/checkpoints/ukiyoe")
    parser.add_argument("--output_dir", type=str, default="models/exported")
    parser.add_argument("--log_dir", type=str, default="logs_v2")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    
    args = parser.parse_args()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    trainer = StyleTransferTrainerV2(
        style_image_path=args.style,
        style_name=args.style_name,
        output_model_dir=args.output_dir,
        content_dir=args.content_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        check_point_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        patience=args.patience
    )
    
    trainer.train()

if __name__ == '__main__':
    main()


