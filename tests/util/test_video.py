import pytest

from neural_ca.train import build_model
from neural_ca.pools import SamplePool
from neural_ca.util.video import make_video, load_video

@pytest.fixture
def video_path():
    return "data/waves.mp4"

@pytest.fixture
def make_pool(example_emoji):
    def _func(pool_size, state_size):
        return SamplePool(
            pool_size=pool_size,
            state_size=state_size,
            emoji=example_emoji
        )
    return _func

@pytest.mark.utils
class TestVideo:
    #TODO: Update the state_size when we add a config
    @pytest.mark.parametrize("steps, state_size",
                             [(128, 16), (256, 16), (512, 16)])
    def test_make_video(self, make_pool, steps, state_size):
        model = build_model()
        pool = make_pool(1, state_size)
        video = make_video(model, pool, steps)
        x, y, *_ = pool.shape
        assert len(video) == steps
        for frame in video:
            assert frame.shape == (x, y, 3)

    @pytest.mark.parametrize("size", [8, 32, 64])
    def test_load_video(self, size, video_path):
        video = load_video(video_path, size=size)
        assert video.shape == (368, size, size, 4)
