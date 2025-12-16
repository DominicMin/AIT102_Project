# Artistic Vision: Fast Arbitrary Image Stylization using TensorFlow Hub & VGG Feature Extraction

## 1. Group Members

| Role | Student Name | Student ID |
| :--- | :--- | :--- |
| **Leader** | [Name] | [ID] |
| **Member** | Lin Jiacheng | DSC2409018 |
| **Member** | [Name] | [ID] |
| **Member** | [Name] | [ID] |
| **Member** | [Name] | [ID] |

## 2. Task Distribution

| Student | Role | Detailed Responsibility |
| :--- | :--- | :--- |
| [Name] | [Role] | [Responsibility] |
| Lin Jiacheng | [Role] | [Responsibility] |
| [Name] | [Role] | [Responsibility] |
| [Name] | [Role] | [Responsibility] |
| [Name] | [Role] | [Responsibility] |

## 3. Model Description

### 3.1 Architectural Overview: Why Artistic Style Transfer

[cite_start]For this project, we implemented a Feed-Forward Neural Network based on the "Fast Arbitrary Image Stylization" architecture[cite: 8].

[cite_start]Unlike the traditional optimization-based approach, which iteratively updates pixel values via backpropagation for every single image (taking minutes per generation), our approach utilizes a pre-trained Encoder-Decoder structure[cite: 9]. [cite_start]This allows for **real-time inference**, stylizing images in milliseconds by passing the data through the network just once[cite: 10]. [cite_start]This architectural decision was critical for enabling the interactive user experience demonstrated in our `main.py`[cite: 11].

#### Notations

| Symbol | Description |
| :---: | :--- |
| $N, H, W$ | Batch Size, The original height and width of content |
| $h_c, w_c$ | The scaled height and width of Content |
| $3$ | Primary color channels |
| AdaIN | Adaptive Instance Normalization |
| $x, y$ | Content feature map, Style embedding |
| $\mu, \sigma$ | The average brightness (mean) and contrast (std) |
| $\gamma, \beta$ | Multiplication factor, Addition factor |

### 3.2 The Inference Pipeline

[cite_start]The following flowchart illustrates the end-to-End inference pipeline implemented in `main.py`[cite: 15]. [cite_start]It details how the raw input images are processed through the Preprocessing Stage, encoded, stylized via AdaIN, and finally reconstructed by the Decoder[cite: 16].

> [cite_start]**Note:** The dimensions are dynamic based on the input image's aspect ratio, with a maximum dimension constraint of 512 pixels[cite: 17].

![Figure 3.1 Neural Style Transfer pipeline](path/to/your/figure_3_1.png)
*Figure 3.1: Neural Style Transfer pipeline*

### 3.3 Tensor Transformation Flow

#### Phase I: Ingestion
[cite_start]Before entering the neural network, raw images must be transformed into compatible Tensors[cite: 21]. In `utils.py`, we implemented a strict ETL pipeline:
* [cite_start]**Dynamic Resizing**: To prevent image distortion, we calculate a scaling factor $scale = max\_dim / long\_dim$ to resize images while preserving their original aspect ratio[cite: 23].
* [cite_start]**Normalization**: Pixel values are normalized from the standard [0, 255] range to a floating-point range of [0.0, 1.0], required for numerical stability in the VGG network[cite: 24].
* [cite_start]**Batch Expansion**: A batch dimension is added (from 3D to 4D tensor) to align with TensorFlow's input requirements[cite: 25].

#### Phase II: Feature Extraction
Our model employs two parallel distinct neural pathways:
1.  [cite_start]**The Content Stream**: Uses a pre-trained **VGG19** network as a fixed feature extractor[cite: 28]. By tapping into intermediate layers (typically `block4_conv1`), we capture high-level semantic structures (shapes) while discarding low-level pixel details. [cite_start]Shallow layers capture exact colors preventing stylization, while deeper layers allow abstract style application[cite: 29, 30].
2.  **The Style Stream**: Uses a specialized **Style Prediction Network**. [cite_start]Instead of spatial features, this network compresses the style image into a 100-dimensional Style Embedding vector $(1, 1, 1, 100)$[cite: 31, 32]. [cite_start]This vector represents the "DNA" of the artistic style (brushstrokes, color distribution)[cite: 33].

#### Phase III: Adaptive Instance Normalization (AdaIN)
The core innovation is the AdaIN layer. It performs a specific operation:
1.  [cite_start]Normalizes content features to remove original style (subtracting mean/std)[cite: 35].
2.  [cite_start]Re-scales and shifts these features using affine parameters from the Style Predictor[cite: 36].
[cite_start]Effectively, this "paints" the content structure with the statistical properties of the style image[cite: 37].

#### Phase IV: Decoding
[cite_start]The final stage is the Decoder Network, trained to invert the VGG Encoder process[cite: 39]. [cite_start]It takes the stylized feature maps and upsamples them back into RGB pixel space[cite: 40]. The output is a $(1, h_c, w_c, 3)$ tensor retaining content structure but exhibiting style textures. [cite_start]Finally, `tensor_to_image` converts this back to PNG format[cite: 41, 42].

**Table 3.2 Overview of tensor transformation flow**

| Stage | Function | Input Shape | Output Shape |
| :--- | :--- | :--- | :--- |
| **I. Ingestion** | `load_img` (utils.py) | $(N, H, W, 3)$ | $(1, h_c, w_c, 3)$ |
| **II. Prediction** | Style Predictor | $(1, h_s, w_s, 3)$ | $(1, 1, 1, 100)$ |
| **III. Fusion** | AdaIN Layer | C: $(1, h', w', d)$ <br> S: $(1, 1, 1, 100)$ | $(1, h', w', d)$ |
| **IV. Decoding** | `tensor_to_image` | $(1, h_c, w_c, 3)$ | $(h_c, w_c, 3)$ |

### 3.4 AdaIN: From Formula to Implementation

[cite_start]The core operation of our style transfer model is defined by the following equation[cite: 46]:

$$\text{AdaIN}(x, y) = \sigma(y) \left( \frac{x - \mu(x)}{\sigma(x)} \right) + \mu(y)$$

AdaIN is essentially a normalization-renormalization process. [cite_start]By replacing the first and second-order statistics (mean and variance) of the content features with those of the style features, we achieve style transfer with high efficiency[cite: 48].

[cite_start]The following diagram visualizes how this equation is executed step-by-step within the TensorFlow computational graph[cite: 49].

![Figure 3.3 The Implementation Chart](path/to/your/figure_3_3.png)
*Figure 3.3: The Implementation Chart*

### 3.5 Loss Function

[cite_start]Although we perform inference using a pre-trained TF-Hub module, understanding the Loss Functions is essential[cite: 52]. [cite_start]The model was trained to minimize a weighted sum of two specific loss functions[cite: 54]:

$$L_{total} = L_c + \lambda L_s$$

**1. Content Loss ($L_c$):**
[cite_start]To ensure the generated image retains the structure of the input photo, we calculate the Euclidean distance ($L_2$ norm) between the features of the Content Image ($f_c$) and the Stylized Image ($f_{gen}$)[cite: 56].

$$L_c = || f(I_c) - f(I_{gen}) ||_2^2$$

[cite_start]*Interpretation:* If the generated image looks like a dog but the input was a cat, this loss spikes, forcing the network to correct the shape[cite: 58].

**2. Style Loss ($L_s$):**
Unlike Content Loss, Style Loss compares statistical distributions. [cite_start]We calculate the difference in Mean ($\mu$) and Standard Deviation ($\sigma$) between the style image features and the generated image features[cite: 59, 60].

$$L_s = || \mu(f_s) - \mu(f_{gen}) ||_2^2 + || \sigma(f_s) - \sigma(f_{gen}) ||_2^2$$

[cite_start]*Interpretation:* This mathematical constraint forces the generated image to have the same "color palette" and "brushstroke complexity" as the artwork[cite: 62].