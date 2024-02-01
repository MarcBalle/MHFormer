import argparse
import os
import glob
import copy

import torch
import numpy as np
from tqdm import tqdm

from model.mhformer import Model
from common.camera import *
from common.skeleton import Skeleton

h36m_skeleton = Skeleton(
    parents=[
        -1,
        0,
        1,
        2,
        3,
        4,
        0,
        6,
        7,
        8,
        9,
        0,
        11,
        12,
        13,
        14,
        12,
        16,
        17,
        18,
        19,
        20,
        19,
        22,
        12,
        24,
        25,
        26,
        27,
        28,
        27,
        30,
    ],
    joints_left=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23],
    joints_right=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31],
)

h36m_skeleton.remove_joints([4, 5, 9, 10, 11, 16, 20, 21, 22, 23, 24, 28, 29, 30, 31])
h36m_skeleton._parents[11] = 8
h36m_skeleton._parents[14] = 8


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-k", "--keypoints", default="cpn_ft_h36m_dbb", type=str, metavar="NAME", help="2D detections to use"
    )

    # Visualization arguments copied from PoseFormer parse_args() function
    parser.add_argument("--viz-subject", type=str, metavar="STR", help="subject to render")
    parser.add_argument("--viz-action", type=str, metavar="STR", help="action to render")
    parser.add_argument("--viz-camera", type=int, default=0, metavar="N", help="camera to render")
    parser.add_argument("--viz-video", type=str, metavar="PATH", help="path to input video")
    parser.add_argument("--viz-skip", type=int, default=0, metavar="N", help="skip first N frames of input video")
    parser.add_argument("--viz-output", type=str, metavar="PATH", help="output file name (.gif or .mp4)")
    parser.add_argument("--viz-export", type=str, metavar="PATH", help="output file name for coordinates")
    parser.add_argument("--viz-bitrate", type=int, default=3000, metavar="N", help="bitrate for mp4 videos")
    parser.add_argument("--viz-no-ground-truth", action="store_true", help="do not show ground-truth poses")
    parser.add_argument("--viz-limit", type=int, default=-1, metavar="N", help="only render first N frames")
    parser.add_argument("--viz-downsample", type=int, default=1, metavar="N", help="downsample FPS by a factor N")
    parser.add_argument("--viz-size", type=int, default=5, metavar="N", help="image size")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    gpu = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    args = parse_arguments()

    mhformer_args = argparse.Namespace()
    mhformer_args.layers, mhformer_args.channel, mhformer_args.d_hid, mhformer_args.frames = 3, 512, 1024, 351
    mhformer_args.pad = (mhformer_args.frames - 1) // 2
    mhformer_args.previous_dir = "checkpoint/pretrained/351"
    mhformer_args.n_joints, mhformer_args.out_joints = 17, 17

    # Reload
    model = Model(mhformer_args).cuda()

    model_dict = model.state_dict()
    # Put the pretrained model of MHFormer in 'checkpoint/pretrained/351'
    model_path = sorted(glob.glob(os.path.join(mhformer_args.previous_dir, "*.pth")))[0]

    pre_dict = torch.load(model_path)
    for name, key in model_dict.items():
        model_dict[name] = pre_dict[name]
    model.load_state_dict(model_dict)

    model.eval()

    # Input in demo/input
    keypoints = np.load(
        os.path.join("demo", "input", args.keypoints),
        allow_pickle=True,
    )["keypoints"]

    # Include batch size (bs x nframes x kp_x x kp_y)
    keypoints = np.expand_dims(keypoints, axis=0)

    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]

    width, height = 960, 540

    # Predicting 3D poses
    print("\nGenerating 3D pose...")
    video_length = keypoints.shape[1]
    output_3d_all = []
    for i in tqdm((range(video_length))):
        # Input frames
        start = max(0, i - mhformer_args.pad)
        end = min(i + mhformer_args.pad, len(keypoints[0]) - 1)

        input_2D_no = keypoints[0][start : end + 1]

        left_pad, right_pad = 0, 0
        if input_2D_no.shape[0] != mhformer_args.frames:
            if i < mhformer_args.pad:
                left_pad = mhformer_args.pad - i
            if i > len(keypoints[0]) - mhformer_args.pad - 1:
                right_pad = i + mhformer_args.pad - (len(keypoints[0]) - 1)

            input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), "edge")

        input_2D = normalize_screen_coordinates(input_2D_no, w=width, h=height)

        input_2D_aug = copy.deepcopy(input_2D)
        input_2D_aug[:, :, 0] *= -1
        input_2D_aug[:, joints_left + joints_right] = input_2D_aug[:, joints_right + joints_left]
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)

        input_2D = input_2D[np.newaxis, :, :, :, :]

        input_2D = torch.from_numpy(input_2D.astype("float32")).cuda()

        N = input_2D.size(0)

        # Prediction all frames
        output_3D_non_flip = model(input_2D[:, 0])
        output_3D_flip = model(input_2D[:, 1])

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        # Select main frame
        output_3D = output_3D[0:, mhformer_args.pad].unsqueeze(1)
        post_out = output_3D[0, 0].cpu().detach().numpy()

        output_3d_all.append(post_out)

    # From here onwards the code is from run_poseformer_fast.py
    # Some needed objects
    keypoints_metadata = {
        "layout_name": "h36m",
        "num_joints": 17,
        "keypoints_symmetry": [[4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]],
    }
    azimuth = np.array(70.0, dtype="float32")

    # Prepare the predictions for visualization
    prediction = np.array(output_3d_all)

    print(f"Prediction shape: {prediction.shape}")

    # For visualization purposes, make all join coordinates relative to the hip joint,
    # being the hip joint the (0, 0, 0). This is because we don't have the camera extrinsics.
    prediction -= np.expand_dims(prediction[:, 0, :], axis=1)

    # Hardcoded (experimentally found) rotation matrix. Only for visualization purposes.
    angle_x = 60 * (np.pi / 180)
    angle_y = 180 * (np.pi / 180)
    angle_z = -30 * (np.pi / 180)

    rot_x = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
    rot_y = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
    rot_z = np.array([[np.cos(angle_z), -np.sin(angle_z), 0], [np.sin(angle_z), np.cos(angle_z), 0], [0, 0, 1]])

    # Hardcoded (experimentally found) translation
    tx = 0
    ty = 0
    tz = 0.75
    t = np.array([tx, ty, tz])

    # Apply extrinsics to predictions
    prediction = (prediction @ rot_x.T @ rot_y.T @ rot_z.T) + t

    anim_output = {"Reconstruction": prediction}

    keypoints = np.squeeze(keypoints)

    # Create visualization
    from common.visualization import render_animation

    render_animation(
        keypoints,
        keypoints_metadata,
        anim_output,
        h36m_skeleton,
        15,
        args.viz_bitrate,
        azimuth,
        args.viz_output,
        limit=args.viz_limit,
        downsample=args.viz_downsample,
        size=args.viz_size,
        input_video_path=args.viz_video,
        viewport=(width, height),
        input_video_skip=args.viz_skip,
    )
