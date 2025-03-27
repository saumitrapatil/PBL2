import os
import time
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy.random import randint
from tensorflow.keras import Input
from numpy import load, zeros, ones
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, UpSampling2D, LeakyReLU, Activation
from tensorflow.keras.layers import Concatenate, Dropout, BatchNormalization

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())


def define_discriminator(image_shape):
    init = RandomNormal(stddev=0.02)

    in_src_image = Input(shape=image_shape)
    in_target_image = Input(shape=image_shape)

    merged = Concatenate()([in_src_image, in_target_image])

    d = Conv2D(64, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(
        merged
    )
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(128, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(256, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(512, (4, 4), strides=(2, 2), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(512, (4, 4), padding="same", kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(1, (4, 4), padding="same", kernel_initializer=init)(d)
    patch_out = Activation("sigmoid")(d)

    model = Model([in_src_image, in_target_image], patch_out)

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, loss_weights=[0.5])
    return model


def define_generator(image_shape=(256, 256, 3)):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)

    g = Conv2D(64, (7, 7), padding="same", kernel_initializer=init)(in_image)
    g = BatchNormalization()(g, training=True)
    g3 = LeakyReLU(alpha=0.2)(g)

    g = Conv2D(128, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(g3)
    g = BatchNormalization()(g, training=True)
    g2 = LeakyReLU(alpha=0.2)(g)

    g = Conv2D(256, (3, 3), strides=(2, 2), padding="same", kernel_initializer=init)(g2)
    g = BatchNormalization()(g, training=True)
    g1 = LeakyReLU(alpha=0.2)(g)

    for _ in range(6):
        g = Conv2D(256, (3, 3), padding="same", kernel_initializer=init)(g1)
        g = BatchNormalization()(g, training=True)
        g = LeakyReLU(alpha=0.2)(g)

        g = Conv2D(256, (3, 3), padding="same", kernel_initializer=init)(g)
        g = BatchNormalization()(g, training=True)

        g1 = Concatenate()([g, g1])

    g = UpSampling2D((2, 2))(g1)
    g = Conv2D(128, (1, 1), kernel_initializer=init)(g)
    g = Dropout(0.5)(g, training=True)
    g = Concatenate()([g, g2])
    g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)

    g = UpSampling2D((2, 2))(g)
    g = Conv2D(64, (1, 1), kernel_initializer=init)(g)
    g = Dropout(0.5)(g, training=True)
    g = Concatenate()([g, g3])
    g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)

    g = Conv2D(3, (7, 7), padding="same", kernel_initializer=init)(g)
    g = BatchNormalization()(g, training=True)
    out_image = Activation("tanh")(g)

    model = Model(in_image, out_image)
    return model


def define_gan(g_model, d_model, image_shape):
    # make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    # define the source image
    in_src = Input(shape=image_shape)
    # connect the source image to the generator input
    gen_out = g_model(in_src)
    # connect the source input and generator output to the discriminator input
    dis_out = d_model([in_src, gen_out])
    # src image as input, generated image and classification output
    model = Model(in_src, [dis_out, gen_out])
    # compile model
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(
        loss=["binary_crossentropy", "mae"], optimizer=opt, loss_weights=[1, 100]
    )
    return model


def load_real_samples(filename):
    # load compressed arrays
    data = load(filename)
    # unpack arrays
    X1, X2 = data["arr_0"], data["arr_1"]
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X2, X1]


def generate_real_samples(dataset, n_samples, patch_shape):
    # unpack dataset
    trainA, trainB = dataset
    # choose random instances
    ix = randint(0, trainA.shape[0], n_samples)
    # retrieve selected images
    X1, X2 = trainA[ix], trainB[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


def generate_fake_samples(g_model, samples, patch_shape):
    # generate fake instance
    X = g_model.predict(samples)
    # create 'fake' class labels (0)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


def summarize_performance(step, g_model, d_model, dataset, n_samples=3):
    # select a sample of input images
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    # plot real source images
    plt.figure(figsize=(14, 14))
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + i)
        plt.axis("off")
        plt.title("Low-Light")
        plt.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples + i)
        plt.axis("off")
        plt.title("Generated")
        plt.imshow(X_fakeB[i])
    # plot real target image
    for i in range(n_samples):
        plt.subplot(3, n_samples, 1 + n_samples * 2 + i)
        plt.axis("off")
        plt.title("Ground Truth")
        plt.imshow(X_realB[i])
    # save plot to file
    filename1 = step_output + "plot_%06d.png" % (step + 1)
    plt.savefig(filename1)
    plt.close()
    # save the generator model
    filename2 = model_output + "gen_model_%06d.h5" % (step + 1)
    g_model.save(filename2)
    # save the discriminator model
    filename3 = model_output + "disc_model_%06d.h5" % (step + 1)
    d_model.save(filename3)
    print("[.] Saved Step : %s" % (filename1))
    print("[.] Saved Model: %s" % (filename2))
    print("[.] Saved Model: %s" % (filename3))


def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=12):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    print("[!] Number of steps {}".format(n_steps))
    print("[!] Saves model/step output at every {}".format(bat_per_epo * 1))
    # manually enumerate epochs
    for i in range(n_steps):
        start = time.time()
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        # summarize performance
        time_taken = time.time() - start
        print(
            "[*] %06d, d1[%.3f] d2[%.3f] g[%06.3f] ---> time[%.2f], time_left[%.08s]"
            % (
                i + 1,
                d_loss1,
                d_loss2,
                g_loss,
                time_taken,
                str(datetime.timedelta(seconds=((time_taken) * (n_steps - (i + 1)))))
                .split(".")[0]
                .zfill(8),
            )
        )
        # summarize model performance
        if (i + 1) % (bat_per_epo * 1) == 0:
            summarize_performance(i, g_model, d_model, dataset)


import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[INFO] GPU is available and being used.")
    except RuntimeError as e:
        print(e)
else:
    print("[WARNING] No GPU found, using CPU!")


from google.colab import drive

drive.mount("/content/drive")

dataset = load_real_samples("/content/drive/MyDrive/PBL2/dataset.npz")
print("Loaded", dataset[0].shape, dataset[1].shape)
image_shape = dataset[0].shape[1:]


d_model = define_discriminator(image_shape)
g_model = define_generator(image_shape)
gan_model = define_gan(g_model, d_model, image_shape)


dir = "/content/drive/MyDrive/PBL2/models"
fileName = "Enhancement Model"
step_output = dir + fileName + "/Step Output/"
model_output = dir + fileName + "/Model Output/"
if fileName not in os.listdir(dir):
    os.mkdir(dir + fileName)
    os.mkdir(step_output)
    os.mkdir(model_output)

train(d_model, g_model, gan_model, dataset, n_batch=12)

import os
import shutil

folder_path = f"/content/modelsEnhancementModel"  # Replace with your actual folder path
zip_filename = "ModelEnhancement.zip"
shutil.make_archive(zip_filename.replace(".zip", ""), "zip", folder_path)

print(f"Zip file created: {zip_filename}")

import os
import shutil
from google.colab import drive


def copy_output_folders_to_drive(source_dir, destination_dir, folder_prefix="output"):
    """
    Copies folders with a specific prefix from a source directory to Google Drive.

    Args:
        source_dir (str): The path to the source directory in Colab.
        destination_dir (str): The path to the destination directory in Google Drive.
        folder_prefix (str): The prefix of the folders to copy.
    """

    # Mount Google Drive
    drive.mount("/content/drive")

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Iterate through the source directory
    for folder_name in os.listdir(source_dir):
        if folder_name.startswith(folder_prefix):
            source_folder = os.path.join(source_dir, folder_name)
            destination_folder = os.path.join(destination_dir, folder_name)

            # Check if it's a directory
            if os.path.isdir(source_folder):
                try:
                    shutil.copytree(source_folder, destination_folder)
                    print(f"Copied '{folder_name}' to Google Drive.")
                except Exception as e:
                    print(f"Error copying '{folder_name}': {e}")
            else:
                print(f"'{folder_name}' is not a directory.")

    print("Copying process completed.")


source_directory = (
    "/content/modelsEnhancementModel"  # Change this to your source directory
)
drive_destination_directory = (
    "/content/drive/MyDrive/model_outputs"  # change this to your desired destination.
)
folder_prefix_to_copy = (
    "modelsEnhancementModel"  # change this if your folders have other prefix.
)

copy_output_folders_to_drive(
    source_directory, drive_destination_directory, folder_prefix_to_copy
)

import os
import shutil
from google.colab import drive

drive.mount("/content/drive")

source_dir = "/content/modelsEnhancementModel/"
destination_dir = "/content/drive/My Drive/Colab_Backup/"

os.makedirs(destination_dir, exist_ok=True)


def copy_files(src, dst):
    for foldername, subfolders, filenames in os.walk(src):
        # Preserve folder structure
        rel_path = os.path.relpath(foldername, src)
        target_folder = os.path.join(dst, rel_path)
        os.makedirs(target_folder, exist_ok=True)

        for filename in filenames:
            if filename.endswith((".h5", ".png")):  # Copy only .h5 and .png files
                src_path = os.path.join(foldername, filename)
                dst_path = os.path.join(target_folder, filename)

                # Copy and preserve metadata
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {src_path} -> {dst_path}")


copy_files(source_dir, destination_dir)

print("Backup completed successfully!")


tf.__version__

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error as mse
import cv2

generator = load_model(
    "/content/drive/MyDrive/Colab_Backup/Model Output/gen_model_004000.h5"
)  # Update the filename accordingly
discriminator = load_model(
    "/content/drive/MyDrive/Colab_Backup/Model Output/disc_model_004000.h5"
)  # Update the filename accordingly

custom_image_path = "/content/testImage.jpeg"  # Change this to your image path


def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, target_size)  # Resize to match training size
    img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
    return np.expand_dims(img, axis=0)  # Add batch dimension


input_image = preprocess_image(custom_image_path)

generated_image = generator.predict(input_image)

generated_image = (generated_image[0] * 127.5 + 127.5).astype(np.uint8)
input_image_disp = ((input_image[0] * 127.5) + 127.5).astype(np.uint8)


def ensure_min_size(image, min_size=7):
    h, w, c = image.shape
    if h < min_size or w < min_size:
        image = cv2.resize(
            image, (max(min_size, w), max(min_size, h))
        )  # Resize while keeping aspect ratio
    return image


input_image_disp = ensure_min_size(input_image_disp)
generated_image = ensure_min_size(generated_image)

real_label = discriminator.predict([input_image, input_image])  # Real image evaluation
fake_label = discriminator.predict(
    [input_image, np.expand_dims(generated_image, axis=0)]
)  # Generated image evaluation

psnr_value = psnr(input_image_disp, generated_image)
ssim_value = ssim(
    input_image_disp, generated_image, win_size=3, channel_axis=-1
)  # Set win_size=3
mse_value = mse(input_image_disp.flatten(), generated_image.flatten())

print("\n[Evaluation Metrics]")
print(f"PSNR: {psnr_value:.2f} dB (Higher is better)")
print(f"SSIM: {ssim_value:.4f} (Higher is better, max=1)")
print(f"MSE: {mse_value:.2f} (Lower is better)")
print(f"Discriminator Output on Real Image: {real_label.mean():.4f}")
print(f"Discriminator Output on Generated Image: {fake_label.mean():.4f}")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(input_image_disp)
plt.axis("off")
plt.title("Input Low-Light Image")

plt.subplot(1, 2, 2)
plt.imshow(generated_image)
plt.axis("off")
plt.title("Generated Enhanced Image")

plt.show()


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error as mse
import cv2

generator = load_model(
    "/content/drive/MyDrive/Colab_Backup/Model Output/gen_model_004000.h5"
)  # Update the filename accordingly
discriminator = load_model(
    "/content/drive/MyDrive/Colab_Backup/Model Output/disc_model_004000.h5"
)  # Update the filename accordingly

custom_image_path = "/content/test2Image.jpeg"  # Change this to your image path


def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, target_size)  # Resize to match training size
    img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
    return np.expand_dims(img, axis=0)  # Add batch dimension


input_image = preprocess_image(custom_image_path)

generated_image = generator.predict(input_image)

generated_image = (generated_image[0] * 127.5 + 127.5).astype(np.uint8)
input_image_disp = ((input_image[0] * 127.5) + 127.5).astype(np.uint8)


def ensure_min_size(image, min_size=7):
    h, w, c = image.shape
    if h < min_size or w < min_size:
        image = cv2.resize(
            image, (max(min_size, w), max(min_size, h))
        )  # Resize while keeping aspect ratio
    return image


input_image_disp = ensure_min_size(input_image_disp)
generated_image = ensure_min_size(generated_image)

real_label = discriminator.predict([input_image, input_image])  # Real image evaluation
fake_label = discriminator.predict(
    [input_image, np.expand_dims(generated_image, axis=0)]
)  # Generated image evaluation

psnr_value = psnr(input_image_disp, generated_image)
ssim_value = ssim(
    input_image_disp, generated_image, win_size=3, channel_axis=-1
)  # Set win_size=3
mse_value = mse(input_image_disp.flatten(), generated_image.flatten())

print("\n[Evaluation Metrics]")
print(f"PSNR: {psnr_value:.2f} dB (Higher is better)")
print(f"SSIM: {ssim_value:.4f} (Higher is better, max=1)")
print(f"MSE: {mse_value:.2f} (Lower is better)")
print(f"Discriminator Output on Real Image: {real_label.mean():.4f}")
print(f"Discriminator Output on Generated Image: {fake_label.mean():.4f}")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(input_image_disp)
plt.axis("off")
plt.title("Input Low-Light Image")

plt.subplot(1, 2, 2)
plt.imshow(generated_image)
plt.axis("off")
plt.title("Generated Enhanced Image")

plt.show()
