import argparse
import os
from typing import Tuple

import event_library as el
import numpy as np
from event_library import utils
from tqdm import tqdm

from experimenting.dataset import HumanCore


def normalized_3sigma(input_img: np.ndarray) -> np.ndarray:
    img = input_img.copy().astype('float')

    sig_img = img[img > 0].std()
    if sig_img < 0.1 / 255:
        sig_img = 0.1 / 255
    numSdevs = 3.0
    range = numSdevs * sig_img

    img[img != 0] *= 255 / range
    img[img < 0] = 0
    img[img > 255] = 255

    return img.astype('uint8')


def constant_count_joint_generator(
    events: np.array,
    joints: np.array,
    num_events: int,
    frame_size: Tuple[int, int],
    n_cameras: int = 4,
) -> np.array:
    """
    Generate constant_count frames and corresponding gt 3D joints labels. 3D joints labels were acquired at 200fps
    """
    event_count_frame = np.zeros((n_cameras, frame_size[0], frame_size[1]), dtype="int")

    start_joint_data_index = 0
    joint_data_fps = 200
    upper_bound = len(joints) * 1 / 200

    for ind, event in enumerate(events):
        y = int(event[0])
        x = int(event[1])
        t = event[2]
        cam = int(event[-1])  # using camera info similar to DHP19

        event_count_frame[cam, x, y] += 1

        if t > upper_bound:
            # Recording ends here
            return

        if (ind + 1) % num_events == 0:

            end_joint_data_index = int(t * joint_data_fps) + 1
            joints_per_frame = np.nanmean(
                joints[start_joint_data_index:end_joint_data_index, :], 0
            )

            for idx in range(n_cameras):
                event_count_frame[idx] = normalized_3sigma(event_count_frame[idx])

            yield event_count_frame, joints_per_frame
            event_count_frame = np.zeros_like(event_count_frame)

            start_joint_data_index = end_joint_data_index

def parse_args():
    parser = argparse.ArgumentParser(
        description="Accumulates events to an event-frame."
    )
    parser.add_argument("--event_files", nargs="+", help="file(s) to convert to output")
    parser.add_argument(
        "--joints_file",
        type=str,
        help="file of .npz joints containing joints data. Generate it using `prepare_data_h3m`",
    )
    parser.add_argument("--output_base_dir", type=str, help="output_dir")
    parser.add_argument(
        "--num_events", type=int, default=30000, help="num events to accumulate"
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    event_files = args.event_files
    joints_file = args.joints_file
    num_events = args.num_events
    data = HumanCore.get_pose_data(joints_file)
    output_base_dir = args.output_base_dir
    hw_info = el.utils.get_hw_property('dvs')
    n_cameras = 4  # Number of parallel cameras

    output_joint_path = os.path.join(output_base_dir, "3d_joints")

    joint_gt = {f"S{s:01d}": {} for s in range(1, 12)}

    cam_index_to_id_map = dict(
        zip(HumanCore.CAMS_ID_MAP.values(), HumanCore.CAMS_ID_MAP.keys())
    )

    for idx in tqdm(range(0, len(event_files), n_cameras)):
        info = HumanCore.get_frame_info(event_files[idx])
        action = info['action']
        action = action.replace("TakingPhoto", "Photo").replace("WalkingDog", "WalkDog")

        if info['subject'] == 11 and action == "Directions":
            print("Discard")
            continue

        if "_ALL" in action:
            print(f"Discard {info}")
            continue

        events = []
        for offset_id in range(0, n_cameras):
            events.append(el.utils.load_from_file(event_files[idx + offset_id]))

        events = [
            np.concatenate(
                [events[index], index * np.ones((len(events[index]), 1))], axis=1
            )
            for index in range(4)
        ]

        events = np.concatenate(events)
        sort_index = np.argsort(events[:, 2])
        events = events[sort_index]
        joints = data[info['subject']][action]['positions']

        frame_and_gt_generator = constant_count_joint_generator(
            events, joints, num_events, hw_info.size
        )

        output_dir = os.path.join(
            output_base_dir, f"S{info['subject']:01d}", f"{action}" + ".{}",
        )

        joints = []
        for ind_frame, event_and_joint_frame in tqdm(enumerate(frame_and_gt_generator)):
            event_frame_per_cams, joint_frame = event_and_joint_frame

            joints.append(joint_frame)
            for id_camera in range(n_cameras):
                cam = cam_index_to_id_map[id_camera]
                os.makedirs(output_dir.format(cam), exist_ok=True)
                np.save(
                    os.path.join(output_dir.format(cam), f"frame{ind_frame:07d}.npy"),
                    event_frame_per_cams[id_camera],
                )

        joint_gt[f"S{info['subject']:01d}"][action] = np.stack(joints)

    np.savez_compressed(output_joint_path, positions_3d=joint_gt)
