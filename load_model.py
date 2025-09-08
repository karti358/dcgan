from model import Generator
from datetime import datetime
import jax
import numpy as np
from pathlib import Path
from flax import nnx
import orbax.checkpoint as ocp

NUM_CHANNELS = 3
LATENT_CHANNEL_SIZE = 100
GENERATOR_FEATURE_SIZE = 128

class ManageCheckpoints:
    def __init__(
        self,
        base_dir: str,
    ):
        self.ckpt_dir = Path(base_dir)
        self.checkpointer = ocp.StandardCheckpointer()

    def save_model(
        self,
        model: nnx.Module
    ) -> None:
        _, state = nnx.split(model)
        
        new_dir = self.ckpt_dir / f"{str(int(datetime.utcnow().timestamp()))}_state"
        self.checkpointer.save(new_dir, state)
        
        return str(new_dir)
        
    def load_model(
        self,
        model: nnx.Module,
        checkpoint: str,
        mesh: jax.sharding.Mesh = None
    ) -> nnx.Module:
        abstract_model = nnx.eval_shape(lambda: model)
        graphdef, abstract_state = nnx.split(abstract_model)
        # nnx.display(abstract_state)

        if mesh:
            abstract_state = jax.tree.map(
                lambda a, s: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=s),
                abstract_state,
                nnx.get_named_sharding(abstract_state, mesh)
            )
        
        restored_state = self.checkpointer.restore(self.ckpt_dir / checkpoint, abstract_state)
        # nnx.display(restored_state)
        
        return nnx.merge(graphdef, restored_state)

MESH = jax.sharding.Mesh(
    devices=np.array(jax.devices()).reshape(4, 2),
    axis_names=('x', "y"),
)

generator = Generator(
    latent_channel_size = LATENT_CHANNEL_SIZE,
    output_channel_size = NUM_CHANNELS,
    feature_map_size = GENERATOR_FEATURE_SIZE,
    rngs = nnx.Rngs(42)
)

g_checkpoint_manager = ManageCheckpoints(
    base_dir="/mnt/e/Class/Projects/dcgan/checkpoints/generator"
)

generator = g_checkpoint_manager.load_model(
    generator,
    "1757344158_state",
    mesh = MESH
)
# nnx.display(generator)