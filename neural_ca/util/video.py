import numpy as np

from neural_ca import util

def make_video(model, image, steps, state_size):
    def process_cell(cell):
        rgb_cell = util.image.to_rgb(cell.numpy()).numpy()
        clipped_cell = np.uint8(rgb_cell.clip(0, 1) * 255)
        unbatched_cell = clipped_cell[0, :, :, :]
        return unbatched_cell

    cell = util.image.make_seeds(image.shape, 1, state_size)
    video = [process_cell(cell)]
    for _ in range(steps - 1):
        cell = model(cell)
        video.append(process_cell(cell))
    return video
