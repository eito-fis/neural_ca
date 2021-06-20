import tensorflow as tf

from neural_ca import util
from neural_ca.pools import SamplePool

class VideoPool(SamplePool):
    def __init__(self, frame_stride, skip_range, *args, **kwargs):
        self.frame_stride = frame_stride
        self.skip_range = skip_range
        super().__init__(*args, **kwargs)

    def sample(self, batch_size):
        sample, sample_idxs = self.build_sample(batch_size)

        # Replace one sampled seed with a fresh seed
        idxs = tf.range(0, self.pool_size)
        replace_idx = tf.random.shuffle(idxs)[:1]
        fresh_seed = self.build_seeds(sample_idxs=replace_idx)
        sample = tf.tensor_scatter_nd_update(
            sample,
            tf.constant([[0]]),
            fresh_seed,
        )
        sample_idxs = tf.tensor_scatter_nd_update(
            sample_idxs,
            tf.constant([[0]]),
            replace_idx,
        )

        target = self.build_target(sample_idxs)
        return sample, target, sample_idxs

    def build_data(self, data):
        return util.video.load_video(data)

    def build_shape(self):
        return self.data.shape[1:]

    def build_pool(self):
        video_length = self.video.shape[0] // self.frame_stride
        assert self.pool_size <= video_length, "Pool size larger than video"
        pool = self.video[:self.pool_size * self.frame_stride:self.frame_stride]
        pool = tf.convert_to_tensor(pool, dtype=tf.float32)
        return pool

    def build_target(self, idxs):
        start, end = self.skip_range
        skip_steps = tf.random.uniform(tf.shape(idxs), start, end, tf.int32)
        target_idxs = idxs + skip_steps
        wrapping_mask = target_idxs >= self.pool_size
        target_idxs -= wrapping_mask * self.pool_size
        target = tf.gather(self.pool, target_idxs, axis=0)
        return target.data

    def build_seeds(self, batch_size=None, sample_idxs=None):
        assert batch_size is not None or sample_idxs is not None, (
            "batch_size or sample_idxs must be specified")
        if sample_idxs is None:
            idxs = tf.range(0, self.pool_size)
            sample_idxs = tf.random.shuffle(idxs)[:batch_size]
        sample_idxs *= self.frame_stride
        seeds = tf.gather(self.video, sample_idxs, axis=0)
        seeds = tf.convert_to_tensor(seeds, dtype=tf.float32)
        return seeds

    @property
    def video(self):
        return self.data
