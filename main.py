from load_model import generator
import jax.numpy as jnp
import numpy as np
import jax
import cv2

NUM_CHANNELS = 3
LATENT_CHANNEL_SIZE = 100
NUM_SAMPLES = 6

keys = jax.random.split(
    jax.random.key(8897),
    NUM_SAMPLES
)

noises = []
for i in range(NUM_SAMPLES):
    noises.append(
        jax.random.normal(
            keys[i],
            shape=(1, 1, 1, LATENT_CHANNEL_SIZE),
            dtype=jnp.float32
        )
    )

noises = jnp.concatenate(
    noises,
    axis = 0
)

images = generator(noises)

mean = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, ...]
std = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, ...]

for i in range(NUM_SAMPLES):
    img = (mean + images[i] * std) * 255.0
    img = img.astype(np.uint8)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    is_successful = cv2.imwrite(f'sample_{i}.png', img)
    if is_successful:
        print("RGB image successfully converted and saved as red_image.png")
    else:
        print("Error: Image failed to save.")