import pytest
import numpy as np

from context import neural_ca
from neural_ca.train import (build_model, build_optimizer, calc_loss, train,
                             main)
from neural_ca.util import load_emoji

@pytest.mark.train
class TestTrain:
    def test_build_model(self):
        build_model()
        assert True

    def test_build_optimizer(self):
        build_optimizer()
        assert True
    
    def test_calc_loss(self):
        def emoji_and_cell(emoji):
            emoji = load_emoji(emoji)
            empty_space = np.zeros(list(emoji.shape[:-1]) + [12])
            cell = np.concatenate((emoji, empty_space), axis=2)
            cell = np.repeat(cell[None, :], 4, axis=0)
            return emoji, cell
        emoji_a, cell_a = emoji_and_cell("😀")
        emoji_b, cell_b = emoji_and_cell("🦎")

        loss_a = calc_loss(cell_a, emoji_a)
        loss_b = calc_loss(cell_b, emoji_b)
        loss_ab = calc_loss(cell_a, emoji_b)
        loss_ba = calc_loss(cell_b, emoji_a)
        assert loss_a == 0
        assert loss_b == 0
        assert loss_ab != 0
        assert loss_ba != 0
    
    def test_main(self):
        main(5)
        assert True
