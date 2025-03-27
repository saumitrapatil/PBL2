import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt

# Load models
diffusion_model = tf.keras.models.load_model("diffusion.keras")
gan_model = tf.keras.models.load_model("gan.h5")

# Load and preprocess input image
def load_image(image_path, target_size=(256, 256)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize to [0,1]
    return img_array

# Post-process image
def post_process(img_array):
    img_array = np.squeeze(img_array, axis=0)  # Remove batch dimension
    img_array = np.clip(img_array * 255.0, 0, 255).astype("uint8")  # Rescale to [0,255]
    return img_array

# Path to input image
image_path = "data/lol_dataset/eval15/low/1.png"

# Step 1: Pass image through diffusion model
input_img = load_image(image_path)
diffused_output = diffusion_model.predict(input_img)

# Step 2: Pass the diffusion output through the GAN
gan_output = gan_model.predict(diffused_output)

# Step 3: Convert and save final output
final_image = post_process(gan_output)

plt.imshow(final_image)
plt.axis("off")
plt.show()

# Save output image
output_path = "output.jpg"
image.array_to_img(final_image).save(output_path)
print(f"Final image saved to {output_path}")
