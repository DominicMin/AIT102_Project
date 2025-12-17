import tensorflow as tf
from tensorflow.keras import layers, Model

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
