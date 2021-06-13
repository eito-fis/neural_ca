import pytest
import numpy as np

from context import neural_ca
from neural_ca.train import build_model
from neural_ca.util.video import make_video

@pytest.mark.utils
class TestVideo:
    #TODO: Update the state_size when we add a config
    @pytest.mark.parametrize("size, steps, state_size",
                             [(32, 128, 16), (64, 256, 16), (128, 512, 16)])
    def test_make_video(self, size, steps, state_size):
        model = build_model()
        img = np.zeros((size, size, 4))
        video = make_video(model, img, steps, state_size)
        assert len(video) == steps
        for frame in video:
            assert frame.shape == (size, size, 3)
