import numpy as np
import torch
from torchvision import io

from visualization import save_tensor2frames, save_tensor2video, play_tensor2video


def cube_division(video_tensor, cube_dim=2, cube_size=16):
    # bs, t, c, h, w ->
    b, t, c, h, w = video_tensor.shape
    cube_t_num = t // cube_dim
    cube_h_num = h // cube_size
    cube_w_num = w // cube_size

    # 0-cube_t_num, 1-cube_dim, 2-c, 3-cube_h_num, 4-cube_size, 5-cube_w_num, 6-cube_size
    cube_group = video_tensor.reshape(b, cube_t_num, cube_dim, c, cube_h_num, cube_size, cube_w_num, cube_size)
    cube_group = cube_group.permute(0, 1, 4, 6, 3, 2, 5, 7)

    return cube_group


def cube_integration(cube_group):
    #  bs, 0-cube_t_num, 1-cube_h_num, 2-cube_w_num, 3-c, 4-cube_dim, 5-cube_size, 6-cube_size
    bs, cube_num_t, cube_num_h, cube_num_w, c, cube_dim, cube_size, cube_size = cube_group.shape
    cube_group = cube_group.permute(0, 1, 5, 4, 2, 6, 3, 7)
    video_tensor = cube_group.reshape(bs, cube_num_t * cube_dim, c, cube_num_h * cube_size, cube_num_w * cube_size)
    return video_tensor


def cube_pos_shuffling(cube_group, shuffling_order):
    if shuffling_order is None or len(shuffling_order) == 0:
        return cube_group

    original_shape = cube_group.shape
    cube_group = torch.flatten(cube_group, start_dim=1, end_dim=3)
    pos_shuffled_cube_group = cube_group[:, shuffling_order, ...]
    pos_shuffled_cube_group = pos_shuffled_cube_group.reshape(original_shape)
    return pos_shuffled_cube_group


def cube_pix_shuffling(cube_group, shuffling_order):
    if shuffling_order is None or len(shuffling_order) == 0:
        return cube_group

    original_shape = cube_group.shape
    cube_group = torch.flatten(cube_group, start_dim=4)
    pix_shuffled_cube_group = cube_group[..., shuffling_order]
    pix_shuffled_cube_group = pix_shuffled_cube_group.reshape(original_shape)
    return pix_shuffled_cube_group


def main():
    cube_dim = 2
    cube_size = 16
    video_path = "video_samples/sample_video.avi"
    key_dict = np.load("key_dicts/key-32-2-16-seed100.npy", allow_pickle=True).item()

    video_tensor, _, _ = io.read_video(video_path, pts_unit='sec', output_format="TCHW")
    video_tensor = video_tensor.unsqueeze(0)

    cube_group = cube_division(video_tensor, cube_dim, cube_size)
    shuffled_cube_group = cube_pos_shuffling(cube_group, key_dict['pos_key'])
    shuffled_cube_group = cube_pix_shuffling(shuffled_cube_group, key_dict['ce_key'])
    encrypted_video = cube_integration(shuffled_cube_group)

    # visualization
    video_cv2format = encrypted_video.squeeze(0).permute(0, 2, 3, 1)

    # play_tensor2video(video_cv2format)
    save_tensor2video(video_cv2format, "video_samples/sample_video_encrypted.avi")
    save_tensor2frames(video_cv2format, "video_samples/frames")


if __name__ == '__main__':
    main()