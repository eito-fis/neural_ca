import numpy as np
import tensorflow as tf

class SamplePool:
    def __init__(self, pool_size, shape, state_size):
        assert pool_size > 0 and state_size > 0
        assert len(shape) == 2 or len(shape) == 3, f"Shape {shape} is wrong length!"
        self.shape = shape
        self.pool_size = pool_size
        self.state_size = state_size
        self.pool = self.build_seeds(pool_size)

    def sample(self, batch_size):
        assert batch_size <= self.pool_size, "Batch size larger than pool size."
        idxs = tf.range(0, self.pool_size)
        sample_idxs = tf.random.shuffle(idxs)[:batch_size]
        sample = tf.gather(self.pool, sample_idxs, axis=0)

        # Replace one sampled seed with a fresh seed
        fresh_seed = self.build_seeds(1)
        sample = tf.tensor_scatter_nd_update(
            sample,
            tf.constant([[0]]),
            fresh_seed,
        )

        return sample, sample_idxs

    def update(self, new_seeds, idxs):
        self.pool = tf.tensor_scatter_nd_update(
            self.pool,
            tf.expand_dims(idxs, axis=-1),
            new_seeds,
        )

    def build_seeds(self, batch_size):
        """ Makes batch of seeds """
        height, width = self.shape[:2]
        seeds = np.zeros([batch_size, height, width, self.state_size])
        seeds[:, height // 2, width // 2, 3:] += 1
        seeds = tf.convert_to_tensor(seeds, dtype=tf.float32)
        return seeds

    def build_target(self):
        pass
