# dcgan_gpu_tf_lowlevel_exp
helper file for gan taining with tensorflow in a low RAM environment

# Low-Level GAN Training on Fashion-MNIST

---

## Task
Experiment with **Generative Adversarial Network (GAN)** on the Fashion-MNIST dataset using **low-level TensorFlow operations**. This helper script provides full control over training loops for both the generator and discriminator.

Key features:

* Low-level training using `tf.function`
* Manual control of training loops (no `.fit()` abstraction)
* Optimized for **Jupyter on Windows with GPU**
* **Memory-safe**: no memory leaks even under GPU + Jupyter + Windows setup
* Real-time loss logging and visual output every N epochs

---

## Description

### 1. Architecture & Functionality

#### 1.1 Generator

A simple MLP (dense-layer) generator that:

* Takes a random noise vector (`codings_size = 30`)
* Outputs a 28×28 grayscale image
* Uses ReLU activations and `sigmoid` on the output
* You can use this for other model architectures as well, just in case of Jupyter do not forget to restart kernel or re-define `@tf.function`-decorated functions

#### 1.2 Discriminator

A basic MLP classifier that:

* Flattens the image
* Uses ReLU activations and a sigmoid output
* Classifies input images as real or fake

#### 1.3 GAN Model

Constructed by stacking the generator and a **frozen** discriminator.

### 2. Training Pipeline

#### 2.1 Dataset

* Dataset: [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)
* Real images scaled to `[0, 1]`
* Batched and prefetched for efficiency

#### 2.2 GAN Training Strategy

* Two separate optimizers (`Adam`) for generator and discriminator
* Discriminator trained with **label smoothing**:

  * Real labels: `0.7–1.2`
  * Fake labels: `0.0–0.1`
* Generator trained via the full GAN model
* Loss: `BinaryCrossentropy`
* Visual outputs generated every 10 epochs
* Execution time and memory use tracked for each run

### 3. Stability & Performance Notes

* Script optimized to run on:

  * **Jupyter Notebook**
  * **Windows**
  * **TensorFlow with GPU enabled**
* Memory diagnostics using `psutil`
* No memory leaks confirmed across 50+ epochs with batch size 32 on NVIDIA RTX hardware

---

## GPU Support

Script auto-detects available GPU:

```bash
GPUs detected: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If TensorFlow is using the CPU, make sure you installed the GPU version or higher:

```bash
pip install tensorflow==2.10.1--upgrade
```

---

## Example Output

Training log (excerpt):

```
Epoch 1/50
After Epoch 1 | D loss: 0.6723 | G loss: 0.8491.
If D loss → 0, discriminator is dominating. 
If G loss is high, generator isn't improving

...

Epoch 10/50
After Epoch 10 | D loss: 0.4652 | G loss: 1.2074.
(Generated samples appear)
```

Visual output (shown every 10 epochs):

---

## Script Breakdown

### Key Components:

| Function                     | Purpose                                                    |
| ---------------------------- | ---------------------------------------------------------- |
| `build_gan()`                | Combines generator and frozen discriminator                |
| `train_discriminator_step()` | One training step for discriminator                        |
| `train_generator_step()`     | One training step for generator via GAN                    |
| `train_gan_low_level()`      | Full GAN training loop with manual logging & visualization |
| `plot_multiple_images()`     | Utility to render generated images                         |
| `print_memory_usage()`       | Logs current RAM usage                                     |

---

## Usage

### 1. Clone or Copy the Script

Save the script as `low_level_gan_train.py` or run directly in a Jupyter Notebook.

### 2. Install Requirements

```bash
pip install tensorflow matplotlib numpy psutil
```

### 3. Run

```bash
 python low_level_gan_train.py
```

Or in Jupyter:

```python
# Paste the script cells and run step by step
```

### 4. Output

* Console logs training progress
* Graphical outputs show generated images
* RAM usage is printed for debugging purposes
* Final time usage is printed on completion

---

## Requirements

```txt
tensorflow>=2.10
numpy
matplotlib
psutil
```

Recommended environment:

* **Python ≥ 3.10**
* **TensorFlow with GPU support**
* **Jupyter (optional but tested and verified)**

---

## Notes

* Discriminator label smoothing helps prevent mode collapse.
* Uses `@tf.function(reduce_retracing=True)` for performance gains without retracing overhead.
* Final generator image quality can be improved by switching to Conv2D layers (DCGAN-style).
* Generator uses `sigmoid` output, so ensure training images are normalized to `[0, 1]`.

---

## Author

**Peteris**
*Trained & tested on TensorFlow 2.10, VS Code Jupyter Notebook, Windows 10, NVIDIA GPU (GTX960M with compute 5.0 - an old and tricky one), CUDA 11.2 + cuDNN 8.1*
**No memory leaks, even on long runs**
