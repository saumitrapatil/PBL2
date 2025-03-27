import os
import cv2
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import plot_model, img_to_array


high_light_train_images_path = "../../data/lol-dataset/lol_dataset/our485/high"
low_light_train_images_path = "../../data/lol-dataset/lol_dataset/our485/low"

high_light_test_images_path = "../../data/lol-dataset/lol_dataset/eval/high"
low_light_test_images_path = "../../data/lol-dataset/lol_dataset/eval/low"

SIZE = 256

def load_images(path, size=224, count=None):
    if count == None:
        files = os.listdir(path)
    else:
        files = os.listdir(path)[:count]
    images = []

    for file in tqdm.tqdm(files):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        img = img.astype("float32") / 255.
        img = img_to_array(img)
        images.append(img)

    images = np.array(images)
    return images

train_low_images = load_images(low_light_train_images_path, size=SIZE)

train_high_images = load_images(high_light_train_images_path, size=SIZE)

test_low_images = load_images(low_light_test_images_path, size=SIZE)

test_high_images = load_images(high_light_test_images_path, size=SIZE)


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))

axes[0].imshow(train_low_images[10])
axes[0].set_title("Low-Light Image")
axes[0].axis("off")

axes[1].imshow(train_high_images[10])
axes[1].set_title("Ground Truth")
axes[1].axis("off")

plt.show()


def down_block(x, filters, kernel_size, apply_batch_normalization=True):
    x = layers.Conv2D(filters, kernel_size, padding="same", strides=2)(x)
    if apply_batch_normalization:
        x = layers.BatchNormalization()(x)

    x = layers.LeakyReLU()(x)
    return x

def up_block(x, skip, filters, kernel_size, dropout=False):
    x = layers.Conv2DTranspose(filters, kernel_size, padding="same", strides=2)(x)
    if dropout:
        x = layers.Dropout(0.1)(x)

    x = layers.LeakyReLU()(x)
    x = layers.concatenate([x, skip])
    return x

def build_model(size):
    inputs = layers.Input(shape=[size, size, 3])

    # Downsampling
    d1 = down_block(inputs, 128, (3, 3), apply_batch_normalization=False)
    d2 = down_block(d1, 128, (3, 3), apply_batch_normalization=False)
    d3 = down_block(d2, 256, (3, 3), apply_batch_normalization=True)
    d4 = down_block(d3, 512, (3, 3), apply_batch_normalization=True)
    d5 = down_block(d4, 512, (3, 3), apply_batch_normalization=True)

    # Upsampling
    u1 = up_block(d5, d4, 512, (3, 3), dropout=False)
    u2 = up_block(u1, d3, 256, (3, 3), dropout=False)
    u3 = up_block(u2, d2, 128, (3, 3), dropout=False)
    u4 = up_block(u3, d1, 128, (3, 3), dropout=False)

    # Final upsampling
    u5 = layers.Conv2DTranspose(64, (3, 3), padding='same', strides=2)(u4)
    u5 = layers.LeakyReLU()(u5)
    u5 = layers.concatenate([u5, inputs])

    # Output layer
    outputs = layers.Conv2D(3, (1, 1), activation='sigmoid', padding='same')(u5)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = build_model(size=SIZE)

model.summary()

plot_model(model, show_shapes=True, show_layer_names=True)

model.compile(
    optimizer = optimizers.Adam(learning_rate=0.001),
    loss = "mean_absolute_error",
    metrics = ["accuracy"]
)


history = model.fit(
    train_low_images,
    train_high_images,
    epochs = 100,
    batch_size = 16,
    validation_data=(test_low_images, test_high_images),
    verbose = 1
)


history_df = pd.DataFrame(history.history)
history_df.head()

plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train", "valid"])
plt.show()

plt.figure()
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["train", "valid"])
plt.show()

def predict_images(test_low, test_high, count=5, size=224):
    for _ in range(count):
        random_idx = np.random.randint(len(test_low))
        predicted = model.predict(test_low[random_idx].reshape(1, size, size, 3), verbose=0)
        predicted = np.clip(predicted, 0.0, 1.0).reshape(size, size, 3)
        
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
        
        axes[0].imshow(test_low[random_idx])
        axes[0].set_title("Low-Light Image")
        axes[0].axis("off")
        
        axes[1].imshow(test_high[random_idx])
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(predicted)
        axes[2].set_title("Enhanced Image")
        axes[2].axis("off")
        
        plt.show()

predict_images(test_low_images, test_high_images, count=5, size=SIZE)
