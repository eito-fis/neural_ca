import pytest
import numpy as np

from neural_ca.train import build_model
from neural_ca.sample_pool import SamplePool
from neural_ca.util.video import make_video, load_video

@pytest.fixture
def video_path():
    return "data/waves.mp4"

@pytest.fixture
def make_pool():
    def _func(pool_size, shape, state_size):
        return SamplePool(
            pool_size=pool_size,
            shape=shape,
            state_size=state_size,
        )
    return _func

@pytest.mark.utils
class TestVideo:
    #TODO: Update the state_size when we add a config
    @pytest.mark.parametrize("size, steps, state_size",
                             [(32, 128, 16), (64, 256, 16), (128, 512, 16)])
    def test_make_video(self, make_pool, size, steps, state_size):
        model = build_model()
        img = np.zeros((size, size, 4))
        pool = make_pool(1, img.shape, state_size)
        video = make_video(model, pool, steps)
        assert len(video) == steps
        for frame in video:
            assert frame.shape == (size, size, 3)

    @pytest.mark.parametrize("size", [8, 32, 64])
    def test_load_video(self, size, video_path):
        video = load_video(video_path, size=size)
        assert video.shape == (368, size, size, 4)
