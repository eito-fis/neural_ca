import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip

from neural_ca import util

def make_video(model, pool, steps):
    def process_cell(cell):
        rgb_cell = util.image.to_rgb(cell.numpy()).numpy()
        clipped_cell = np.uint8(rgb_cell.clip(0, 1) * 255)
        unbatched_cell = clipped_cell[0, :, :, :]
        return unbatched_cell

    cell = pool.build_seeds(1)
    video = [process_cell(cell)]
    for _ in range(steps - 1):
        cell = model(cell)
        video.append(process_cell(cell))
    return video

def load_video(path, size=64):
    def process_frame(frame):
        frame = Image.fromarray(frame)
        frame = frame.convert("RGBA")
        frame = util.image.process_image(frame, size=size)
        frame = frame[:, :, :4]
        return frame

    with VideoFileClip(path, audio=False, target_resolution=(size, size)) as video:
        # Each frame is [height, width, 4]
        frames = [process_frame(frame) for frame in video.iter_frames()]
    # Now [n_frames, height, width, 3]
    frames = np.stack(frames, axis=0)
    return frames
