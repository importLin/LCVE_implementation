import os

import cv2
import numpy as np


def save_tensor2video(video_tensor, output_path, fps=30):
    if video_tensor.shape[-1] != 3:
        raise ValueError('the last dimension of input expected to be 3')

    frame_group = video_tensor.numpy()

    # XVID Encoder
    fourcc = cv2.VideoWriter_fourcc(*'I420')
    frame_size = (frame_group.shape[-3], frame_group.shape[-2])
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for i in range(frame_group.shape[0]):
        frame = frame_group[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    video_writer.release()


def save_tensor2frames(video_tensor, output_root, end_index=8):
    if video_tensor.shape[-1] != 3:
        raise ValueError('the last dimension of input expected to be 3')
    video_array = video_tensor[:end_index].numpy()
    for i in range(video_array.shape[0]):

        frame = video_array[i].astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        output_path = os.path.join(output_root, f"frame{i}.png")
        cv2.imwrite(output_path, frame)


def play_tensor2video(video_tensor, delay=64):
    if video_tensor.shape[-1] != 3:
        raise ValueError('the last dimension of input expected to be 3')
    cv2.namedWindow('Video Playback', cv2.WINDOW_NORMAL)
    video_array = video_tensor.numpy()
    for frame in video_array:

        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Video Playback', frame)

        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    # 释放资源并关闭窗口
    cv2.destroyAllWindows()
