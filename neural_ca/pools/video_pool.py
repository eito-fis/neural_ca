import numpy as np
import tensorflow as tf

from neural_ca import util
from neural_ca.pools import SamplePool

class VideoPool(SamplePool):
    def __init__(self, frame_stride, skip_range, *args, **kwargs):
        self.frame_stride = frame_stride
        self.skip_range = skip_range
        super().__init__(*args, **kwargs)
        self.target_idxs = tf.range(self.pool_size)

    def sample(self, batch_size):
        sample, sample_idxs = self.build_sample(batch_size)

        num_replace = 1
        idxs = tf.range(0, self.pool_size)
        replace_idx = tf.random.shuffle(idxs)[:num_replace]
        fresh_seed = self.build_seeds(sample_idxs=replace_idx)
        sample = tf.tensor_scatter_nd_update(
            sample,
            tf.constant(tf.range(num_replace))[:, None],
            fresh_seed,
        )
        sample_idxs = tf.tensor_scatter_nd_update(
            sample_idxs,
            tf.constant(tf.range(num_replace))[:, None],
            replace_idx,
        )

        target = self.build_target(sample_idxs)
        return sample, target, sample_idxs

    def build_data(self, data):
        video = util.video.load_video(data)
        video = video[::self.frame_stride]
        repeat = self.pool_size // video.shape[0]
        if repeat >= 1:
            video = np.repeat(video, repeat + 1, axis=0)
        return video

    def build_shape(self):
        return self.data.shape[1:]

    def build_pool(self):
        pool = self.video[:self.pool_size]
        pool = tf.convert_to_tensor(pool, dtype=tf.float32)
        pool = self._add_state(pool, self.state_size)
        return pool

    def build_target(self, sample_idxs):
        target_idxs = self.build_target_idxs(sample_idxs)
        target = tf.gather(self.video, target_idxs, axis=0)
        return util.image.to_rgba(target)

    def build_target_idxs(self, sample_idxs):
        batch_targets = tf.gather(self.target_idxs, sample_idxs, axis=0)
        start, end = self.skip_range
        skip_steps = tf.random.uniform(tf.shape(batch_targets), start, end, tf.int32)
        target_idxs = batch_targets + skip_steps
        wrapping_mask = tf.cast(target_idxs >= self.pool_size, tf.int32)
        target_idxs -= wrapping_mask * self.pool_size

        self.target_idxs = tf.tensor_scatter_nd_update(
            self.target_idxs,
            sample_idxs[:, None],
            target_idxs
        )

        return target_idxs

    def build_seeds(self, batch_size=None, sample_idxs=[0]):
        assert batch_size is not None or sample_idxs is not None, (
            "batch_size or sample_idxs must be specified")
        if sample_idxs is None:
            idxs = tf.range(0, self.pool_size)
            sample_idxs = tf.random.shuffle(idxs)[:batch_size]
        seeds = tf.gather(self.video, sample_idxs, axis=0)
        seeds = tf.convert_to_tensor(seeds, dtype=tf.float32)
        seeds = self._add_state(seeds, self.state_size)
        return seeds

    @staticmethod
    def _add_state(batch, state_size):
        batch_shape = tf.shape(batch)
        length, height, width, depth = batch_shape
        state = tf.ones((length, height, width, state_size - depth))
        batch = tf.concat((batch, state), axis=-1)
        return batch

    @property
    def video(self):
        return self.data
