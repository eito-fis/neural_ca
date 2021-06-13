import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip

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

def load_video(path, size=64):
    def process_frame(frame):
        # Add fake alpha channel to normal images
        # [height, width]
        alpha = np.ones_like(frame[:, :, 0])
        # [height, width, 4]
        frame = np.concatenate((frame, alpha[:, :, None]), axis=2)
        frame = Image.fromarray(frame, mode="RGBA")
        frame = util.image.process_image(frame, size=size)

    with VideoFileClip(path, audio=False) as video:
        # Each frame is [height, width, 4]
        frames = [process_frame(frame) for frame in video.iter_frames()]
    # Now [n_frames, height, width, 3]
    frames = np.stack(frames, axis=0)
    return frames
