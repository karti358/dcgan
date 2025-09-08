
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from flax import nnx
import flax
import matplotlib.pyplot as plt
from collections.abc import Iterable
from typing import Literal, Union, Sequence, Callable, Any
from collections import defaultdict

def normal_with_mean(mean: float = 0.0, stddev: float = 1.0, dtype=jnp.float32):
    """Return an initializer with specified mean and stddev."""
    def init(key: jax.Array, shape: Sequence[int], dtype=dtype):
        return mean + stddev * jax.random.normal(key, shape, dtype)
    return init


class ConvTranspose2D(nnx.Module):
    def __init__(
        self,
        **kwargs
    ):
        assert isinstance(kwargs["kernel_size"], Iterable)
        assert len(kwargs["kernel_size"]) == 2
        
        self.conv_layer = nnx.ConvTranspose(**kwargs)

    def __call__(self, inputs):
        return self.conv_layer(inputs)


class Generator(nnx.Module):
    def __init__(
        self,
        latent_channel_size: int,
        output_channel_size: int,
        feature_map_size: int,
        rngs: nnx.Rngs
    ):
        init_fn = nnx.initializers.normal(stddev=0.02)
        scale_init_fn = normal_with_mean(mean = 1.0, stddev = 0.02)
        bias_init_fn = nnx.initializers.zeros_init()

        #Layer 1
        self.g_conv_1 = ConvTranspose2D(
            in_features = latent_channel_size,
            out_features = feature_map_size * 8,
            kernel_size = (4, 4),
            padding = "VALID",
            strides = 1,
            use_bias=False,
            kernel_init = nnx.with_partitioning(
                init_fn,
                (None, None, "y", "x")
            ),
            rngs = rngs
        )
        self.g_batch_1 = nnx.BatchNorm(
            num_features = feature_map_size * 8,
            axis = -1,
            bias_init = nnx.with_partitioning(
                bias_init_fn,
                ("x",)
            ),
            scale_init = nnx.with_partitioning(
                scale_init_fn,
                ("x",)
            ),
            rngs = rngs
        )

        #Layer 2
        self.g_conv_2 = ConvTranspose2D(
            in_features = feature_map_size * 8,
            out_features = feature_map_size * 4,
            kernel_size = (4, 4),
            padding = 2,
            strides = 2,
            use_bias=False,
            kernel_init = nnx.with_partitioning(
                init_fn,
                (None, None, "y", "x")
            ),
            rngs = rngs
        )
        self.g_batch_2 = nnx.BatchNorm(
            num_features = feature_map_size * 4,
            axis = -1,
            bias_init = nnx.with_partitioning(
                bias_init_fn,
                ("x",)
            ),
            scale_init = nnx.with_partitioning(
                scale_init_fn,
                ("x",)
            ),
            rngs = rngs
        )

        #Layer 3
        self.g_conv_3 = ConvTranspose2D(
            in_features = feature_map_size * 4,
            out_features = feature_map_size * 2,
            kernel_size = (4, 4),
            padding = 2,
            strides = 2,
            use_bias=False,
            kernel_init = nnx.with_partitioning(
                init_fn,
                (None, None, "y", "x")
            ),
            rngs = rngs
        )
        self.g_batch_3 = nnx.BatchNorm(
            num_features = feature_map_size * 2,
            axis = -1,
            bias_init = nnx.with_partitioning(
                bias_init_fn,
                ("x",)
            ),
            scale_init = nnx.with_partitioning(
                scale_init_fn,
                ("x",)
            ),
            rngs = rngs
        )

        #Layer 4
        self.g_conv_4 = ConvTranspose2D(
            in_features = feature_map_size * 2,
            out_features = feature_map_size,
            kernel_size = (4, 4),
            padding = 2,
            strides = 2,
            use_bias=False,
            kernel_init = nnx.with_partitioning(
                init_fn,
                (None, None, "y", "x")
            ),
            rngs = rngs
        )
        self.g_batch_4 = nnx.BatchNorm(
            num_features = feature_map_size,
            axis = -1,
            bias_init = nnx.with_partitioning(
                bias_init_fn,
                ("x",)
            ),
            scale_init = nnx.with_partitioning(
                scale_init_fn,
                ("x",)
            ),
            rngs = rngs
        )
        
        #Layer 5
        self.g_conv_5 = ConvTranspose2D(
            in_features = feature_map_size,
            out_features = feature_map_size // 2,
            kernel_size = (4, 4),
            padding = 2,
            strides = 2,
            use_bias=False,
            kernel_init = nnx.with_partitioning(
                init_fn,
                (None, None, "y", "x")
            ),
            rngs = rngs
        )
        self.g_batch_5 = nnx.BatchNorm(
            num_features = feature_map_size // 2,
            axis = -1,
            bias_init = nnx.with_partitioning(
                bias_init_fn,
                ("x",)
            ),
            scale_init = nnx.with_partitioning(
                scale_init_fn,
                ("x",)
            ),
            rngs = rngs
        )

        #Layer 6
        self.g_conv_6 = ConvTranspose2D(
            in_features = feature_map_size // 2,
            out_features = output_channel_size,
            kernel_size = (4, 4),
            padding = 2,
            strides = 2,
            use_bias=False,
            kernel_init = nnx.with_partitioning(
                init_fn,
                (None, "x", "y", None)
            ),
            rngs = rngs
        )

    def __call__(self, inputs):
        x = nnx.relu(self.g_batch_1(self.g_conv_1(inputs)))
        x = nnx.relu(self.g_batch_2(self.g_conv_2(x)))
        x = nnx.relu(self.g_batch_3(self.g_conv_3(x)))
        x = nnx.relu(self.g_batch_4(self.g_conv_4(x)))
        x = nnx.relu(self.g_batch_5(self.g_conv_5(x)))
        x = nnx.tanh(self.g_conv_6(x))
        return x


class Conv2D(nnx.Module):
    def __init__(
        self,
        **kwargs
    ):
        assert isinstance(kwargs["kernel_size"], Iterable)
        assert len(kwargs["kernel_size"]) == 2
        
        self.conv_layer = nnx.Conv(**kwargs)

    def __call__(self, inputs):
        return self.conv_layer(inputs)


class Discriminator(nnx.Module):
    def __init__(
        self,
        input_channel_size: int,
        feature_map_size: int,
        rngs: nnx.Rngs
    ):
        init_fn = nnx.initializers.normal(stddev=0.02)
        scale_init_fn = normal_with_mean(mean = 1.0, stddev = 0.02)
        bias_init_fn = nnx.initializers.zeros_init()

        # Layer 1
        self.d_conv_1 = Conv2D(
            in_features = input_channel_size,
            out_features = feature_map_size // 2,
            kernel_size = (4, 4),
            padding = 1,
            strides = 2,
            use_bias=False,
            kernel_init = nnx.with_partitioning(
                init_fn,
                (None, "x", None, "y")
            ),
            rngs = rngs
        )

        #Layer 2
        self.d_conv_2 = Conv2D(
            in_features = feature_map_size // 2,
            out_features = feature_map_size,
            kernel_size = (4, 4),
            padding = 1,
            strides = 2,
            use_bias=False,
            kernel_init = nnx.with_partitioning(
                init_fn,
                (None, None, "y", "x")
            ),
            rngs = rngs
        )
        self.d_batch_2 = nnx.BatchNorm(
            num_features = feature_map_size,
            axis = -1,
            bias_init = nnx.with_partitioning(
                bias_init_fn,
                ("x",)
            ),
            scale_init = nnx.with_partitioning(
                scale_init_fn,
                ("x",)
            ),
            rngs = rngs
        )

        #Layer 3
        self.d_conv_3 = Conv2D(
            in_features = feature_map_size,
            out_features = feature_map_size * 2,
            kernel_size = (4, 4),
            padding = 1,
            strides = 2,
            use_bias=False,
            kernel_init = nnx.with_partitioning(
                init_fn,
                (None, None, "y", "x")
            ),
            rngs = rngs
        )
        self.d_batch_3 = nnx.BatchNorm(
            num_features = feature_map_size * 2,
            axis = -1,
            bias_init = nnx.with_partitioning(
                bias_init_fn,
                ("x",)
            ),
            scale_init = nnx.with_partitioning(
                scale_init_fn,
                ("x",)
            ),
            rngs = rngs
        )

        #Layer 4
        self.d_conv_4 = Conv2D(
            in_features = feature_map_size * 2,
            out_features = feature_map_size * 4,
            kernel_size = (4, 4),
            padding = 1,
            strides = 2,
            use_bias=False,
            kernel_init = nnx.with_partitioning(
                init_fn,
                (None, None, "y", "x")
            ),
            rngs = rngs
        )
        self.d_batch_4 = nnx.BatchNorm(
            num_features = feature_map_size * 4,
            axis = -1,
            bias_init = nnx.with_partitioning(
                bias_init_fn,
                ("x",)
            ),
            scale_init = nnx.with_partitioning(
                scale_init_fn,
                ("x",)
            ),
            rngs = rngs
        )

        #Layer 5
        self.d_conv_5 = Conv2D(
            in_features = feature_map_size * 4,
            out_features = feature_map_size * 8,
            kernel_size = (4, 4),
            padding = 1,
            strides = 2,
            use_bias=False,
            kernel_init = nnx.with_partitioning(
                init_fn,
                (None, None, "y", "x")
            ),
            rngs = rngs
        )
        self.d_batch_5 = nnx.BatchNorm(
            num_features = feature_map_size * 8,
            axis = -1,
            bias_init = nnx.with_partitioning(
                bias_init_fn,
                ("x",)
            ),
            scale_init = nnx.with_partitioning(
                scale_init_fn,
                ("x",)
            ),
            rngs = rngs
        )

        #Layer 6
        self.d_conv_6 = Conv2D(
            in_features = feature_map_size * 8,
            out_features = 1,
            kernel_size = (4, 4),
            padding = "VALID",
            strides = 1,
            use_bias=False,
            kernel_init = nnx.with_partitioning(
                init_fn,
                (None, None, "y", "x")
            ),
            rngs = rngs
        )
        
    def __call__(self, inputs):
        x = nnx.leaky_relu(self.d_conv_1(inputs), negative_slope = 0.2)
        x = nnx.leaky_relu(self.d_batch_2(self.d_conv_2(x)), negative_slope=0.2)
        x = nnx.leaky_relu(self.d_batch_3(self.d_conv_3(x)), negative_slope=0.2)
        x = nnx.leaky_relu(self.d_batch_4(self.d_conv_4(x)), negative_slope=0.2)
        x = nnx.leaky_relu(self.d_batch_5(self.d_conv_5(x)), negative_slope=0.2)
        x = self.d_conv_6(x)
        return x
