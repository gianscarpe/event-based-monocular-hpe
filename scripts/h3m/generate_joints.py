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
    events: np.array, joints: np.array, num_events: int, frame_size: Tuple[int, int]
) -> np.array:
    """
    Generate constant_count frames and corresponding gt 3D joints labels. 3D joints labels were acquired at 200fps
    """
    event_count_frame = np.zeros((frame_size[0], frame_size[1], 1), dtype="int")
    start_joint_data_index = 0
    start_time = events[0][2]
    joint_data_fps = 200

    for ind, event in enumerate(events):
        y = int(event[0])
        x = int(event[1])
        t = event[2]
        event_count_frame[x, y] += 1
        if (ind + 1) % num_events == 0:

            end_joint_data_index = (
                start_joint_data_index + int((t - start_time) * joint_data_fps) + 1
            )
            joints_per_frame = np.nanmean(
                joints[start_joint_data_index:end_joint_data_index, :], 0
            )

            yield normalized_3sigma(event_count_frame), joints_per_frame
            event_count_frame = np.zeros_like(event_count_frame)

            start_time = t
            start_joint_data_index = end_joint_data_index


def parse_args():
    parser = argparse.ArgumentParser(
        description="Accumulates events to an event-frame."
    )
    parser.add_argument("--event_files", nargs="+", help="file(s) to convert to output")
    parser.add_argument("--joints_file", type=str, help="file of .npz joints")
    parser.add_argument("--output_base_dir", type=str, help="output_dir")
    parser.add_argument(
        "--num_events", type=int, default=5000, help="num events to accumulate"
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    event_files = args.event_files
    joints_file = args.joints_file
    data = HumanCore.get_pose_data(joints_file)
    output_base_dir = args.output_base_dir
    hw_info = el.utils.get_hw_property('dvs')
    joint_gt = {}
    output_joint_path = os.path.join(output_base_dir, "3d_joints")

    for event_file in event_files:
        events = el.utils.load_from_file(event_file)
        data = HumanCore.get_pose_data(joints_file)
        info = HumanCore.get_frame_info(event_file)
        if info['subject'] == 11 and info['action'] == "Directions":
            print("Discard")
            continue

        joints = data[info['subject']][info['action']]['positions']

        cam_index_to_id_map = dict(
            zip(HumanCore.CAMS_ID_MAP.values(), HumanCore.CAMS_ID_MAP.keys())
        )
        frame_and_gt_generator = constant_count_joint_generator(
            events, joints, 7500, hw_info.size
        )
        output_dir = os.path.join(
            output_base_dir,
            f"S{info['subject']:01d}",
            f"{info['action']}.{cam_index_to_id_map[info['cam']]}",
        )

        os.makedirs(output_dir, exist_ok=True)
        joints = []
        for ind, event_and_joint_frame in tqdm(enumerate(frame_and_gt_generator)):
            event_frame, joint_frame = event_and_joint_frame
            joints.append(joint_frame)
            np.save(os.path.join(output_dir, f"frame{ind:07d}.npy"), event_frame)

        joint_gt[f"S{info['subject']:01d}"] = {}
        joint_gt[f"S{info['subject']:01d}"][info['action']] = np.stack(joints)

    np.savez_compressed(output_joint_path, positions_3d=joint_gt)
