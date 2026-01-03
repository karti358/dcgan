import argparse
from load_model import generator
import jax.numpy as jnp
import numpy as np
import jax
import cv2
import time
import os
import random
from uuid import uuid4

def main(output_path, seed):
    # Ensure output_path has a valid extension
    if not os.path.splitext(output_path)[1]:
        output_path += ".png"

    NUM_CHANNELS = 3
    LATENT_CHANNEL_SIZE = 100

    key = jax.random.key(seed)

    noise = jax.random.normal(
        key,
        shape=(1, 1, 1, LATENT_CHANNEL_SIZE),
        dtype=jnp.float32
    )

    images = generator(noise)

    mean = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, ...]
    std = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, ...]

    img = (mean + images[0] * std) * 255.0
    img = img.astype(np.uint8)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    is_successful = cv2.imwrite(output_path, img)
    if is_successful:
        print(f"Image successfully saved to {output_path}")
    else:
        print("Error: Image failed to save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate an image using a DCGAN.')
    parser.add_argument(
        '--output',
        type=str,
        default= f"./outputs/{uuid4().hex}.png",
        help='Output file path for the generated image.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default = random.randint(1, 10000000),
        help='Random seed for key generation.'
    )
    args = parser.parse_args()
    main(args.output, args.seed)