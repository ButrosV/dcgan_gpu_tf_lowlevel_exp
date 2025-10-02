import tensorflow as tf
import numpy as np
import psutil
import matplotlib.pyplot as plt
import time  # <-- Added for timing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO (0), WARNING (1), and show ERROR (2) and show FATAL (3)

# NOTE: THIS CODE RUNS FINE AS A SCRIPT AND IN JUPYTER+WINDOWS+GPU (NO MEMORY LEAKS)

#UTILITIES :

# --- Plot function for visual confirmation ---
def plot_multiple_images(images, n_cols=None):
    """
    Display multiple images in a grid layout using matplotlib.

    :param images: Iterable of images (e.g., numpy arrays or tensors) to be plotted.
    :param n_cols: Optional int, number of columns in the grid. Defaults to 5 if not provided.
    :return: None. Displays the images in a matplotlib figure with axes turned off.
    """
    n_cols = n_cols or 5
    n_rows = (len(images) + n_cols - 1) // n_cols
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images[:n_cols * n_rows]):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")


def print_memory_usage(stage=""):
    """
    Print current memory usage of the Python process.
    :param stage: str, optional label to identify the stage of execution when memory is checked.
    :return: None
    """
    process = psutil.Process()
    mem = process.memory_info().rss / 1e6  # Convert bytes to MB
    print(f"[Memory] {stage}: {mem:.2f} MB")


# MODEL BUILD function to separate generator and discriminator training
def build_gan(generator, discriminator, codings_size):
    """
    Build a GAN model by connecting a frozen discriminator to the generator.

    :param generator: tf.keras.Model, the generator model that generates fake images from noise.
    :param discriminator: tf.keras.Model, the discriminator model that classifies images as real or fake.
                          This model's weights are frozen (non-trainable) in the GAN model.
    :param codings_size: int, dimensionality of the noise input vector to the generator.
    :return: A tf.keras.Model representing the combined GAN model where the generator's output
             is fed into the frozen discriminator.
    """
    # Freeze discriminator for the GAN model
    discriminator.trainable = False
    gan_input = tf.keras.Input(shape=(codings_size,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan_model = tf.keras.Model(gan_input, gan_output)
    return gan_model


# MODEL TRAIN FUNCTIONS
@tf.function(reduce_retracing=True)
def train_discriminator_step(generator, discriminator, batch_size, codings_size, real_images,
                              loss_fn, optimizer):
    """
    Perform one training step for the discriminator.
    
    This function is decorated with `@tf.function(reduce_retracing=True)` to compile it into a
    TensorFlow graph for performance improvements. The `reduce_retracing=True` argument
    instructs TensorFlow to minimize retracing of the function when called with inputs of different shapes or types,
    which is useful when training multiple models or datasets with varying shapes in the same runtime.

    :param generator: tf.keras.Model, generator used to produce fake images from noise.
    :param discriminator: tf.keras.Model, discriminator to be trained.
    :param batch_size: int, number of samples per batch.
    :param codings_size: int, dimensionality of the noise input vector.
    :param real_images: tf.Tensor, batch of real images from the dataset.
    :param loss_fn: loss function to compute discriminator loss (e.g., BinaryCrossentropy).
    :param optimizer: tf.keras.optimizers.Optimizer used to update discriminator weights.
    :return: scalar tensor representing the discriminator loss for the step.
    """
    noise = tf.random.normal([batch_size, codings_size])
    generated_images = generator(noise, training=True)  # default training=True
    X_fake_and_real = tf.concat([generated_images, real_images], axis=0)
    # y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)  # original
    # alternative y1 w smoothing as discriminator was too strong
    real_labels = tf.random.uniform((batch_size, 1), minval=0.7, maxval=1.2)
    fake_labels = tf.random.uniform((batch_size, 1), minval=0.0, maxval=0.1)
    y1 = tf.concat([fake_labels, real_labels], axis=0)

    with tf.GradientTape() as tape:
        y_pred = discriminator(X_fake_and_real, training=True)
        loss = loss_fn(y1, y_pred)
    gradients = tape.gradient(loss, discriminator.trainable_variables)
    optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    return loss


@tf.function(reduce_retracing=True)
def train_generator_step(generator, gan_model, batch_size, codings_size,
                          loss_fn, optimizer):
    """
    Perform one training step for the generator via the combined GAN model.
    
    This function is decorated with `@tf.function(reduce_retracing=True)` to compile it into a
    TensorFlow graph for performance improvements.

    :param generator: tf.keras.Model, the generator model to be trained.
    :param gan_model: tf.keras.Model, combined GAN model with frozen discriminator.
    :param batch_size: int, number of samples per batch.
    :param codings_size: int, dimensionality of the noise input vector.
    :param loss_fn: loss function to compute generator loss (e.g., BinaryCrossentropy).
    :param optimizer: tf.keras.optimizers.Optimizer used to update generator weights.
    :return: scalar tensor representing the generator loss for the step.
    """
    noise = tf.random.normal([batch_size, codings_size])
    y2 = tf.constant([[1.]] * batch_size)

    with tf.GradientTape() as tape:
        # gan_model includes generator and frozen discriminator
        y_pred = gan_model(noise, training=True)
        loss = loss_fn(y2, y_pred)

    gradients = tape.gradient(loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    return loss


def train_gan_low_level(generator, discriminator, dataset, gan_model, batch_size, codings_size, n_epochs):
    """
    Train the GAN model using low-level TensorFlow operations over multiple epochs.
    :param generator: tf.keras.Model, the generator model to be trained.
    :param discriminator: tf.keras.Model, the discriminator model to be trained.
    :param dataset: tf.data.Dataset, dataset of real images batched and prefetched for training.
    :param gan_model: tf.keras.Model, combined GAN model with frozen discriminator for generator training.
    :param batch_size: int, number of samples per batch.
    :param codings_size: int, dimensionality of the noise input vector.
    :param n_epochs: int, number of epochs to train the GAN.
    :return: None
    """
    # Define optimizers and loss function
    bce = tf.keras.losses.BinaryCrossentropy()

    # experiments with Adam as original RMSprop had issues with too weak generator
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)  # beta_1 (momentum  coef via moving averages of gradients) = 0.9 is default, but GANs often better w reduced 0.5 momentum
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")  # extra code

        for X_batch in dataset:
            discriminator_loss = train_discriminator_step(generator, discriminator, batch_size, codings_size, X_batch,
                                                          bce, discriminator_optimizer)
        
            generator_loss = train_generator_step(generator, gan_model, batch_size, codings_size,
                                                  bce, generator_optimizer)

        print(f"After Epoch {epoch+1} | D loss: {discriminator_loss:.4f} | G loss: {generator_loss:.4f}.\
              \nIf D loss → 0, discriminator is dominating. \nIf G loss is high, generator isn't improving")
        if (epoch + 1) % 10 == 0:
            noise = tf.random.normal([batch_size, codings_size])
            generated_images = generator(noise, training=False)
            plot_multiple_images(generated_images.numpy(), 8)
            plt.show(block=False)  # code continues running without waiting for the window to close manually, requires plt.pause(some fraction of second) for rendering
            plt.pause(2)   # Show for 2 seconds
            plt.close('all')


# --- Check available RAM ---
print(f"Available RAM: {psutil.virtual_memory().available / 1e6:.2f} MB")


# Detect available GPUs
gpus = tf.config.list_physical_devices('GPU')
print("GPUs detected:", gpus)

# --- Start timing ---
start_time = time.time()

# loads, scales, and splits the fashion MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

# --- Build GAN ---
codings_size = 30
Dense = tf.keras.layers.Dense

generator = tf.keras.Sequential([
    Dense(100, activation="relu", kernel_initializer="he_normal"),
    Dense(150, activation="relu", kernel_initializer="he_normal"),
    Dense(28 * 28, activation="sigmoid"),
    tf.keras.layers.Reshape([28, 28])
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    Dense(150, activation="relu", kernel_initializer="he_normal"),
    Dense(100, activation="relu", kernel_initializer="he_normal"),
    Dense(1, activation="sigmoid")
])

gan_model = build_gan(generator, discriminator, codings_size)  # stores discriminator.trainable = False
discriminator.trainable = True # flip back discriminator.trainable for discriminator training

# Prepare dataset
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

# --- Start training ---
train_gan_low_level(generator, discriminator, dataset, gan_model, batch_size, codings_size, n_epochs=50)

# --- End timing and print duration ---
end_time = time.time()
elapsed = end_time - start_time
mins, secs = divmod(elapsed, 60)

print(f"\n✅ Training completed without crashing.")
print(f"🕒 Elapsed time: {int(mins)} minutes, {secs:.2f} seconds")
