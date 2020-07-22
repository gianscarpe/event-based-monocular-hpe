import cv2
import os
import numpy as np

from tqdm import tqdm

__all__ = ['MPII_extractor', 'NTU_extractor']


class NTU_extractor:
    def _get_ntu_video_files(input_dir):
        video_files = []
        for root, dirs, files in os.walk(input_dir):
            for f in files:

                if f.endswith('.avi'):
                    video_files.append(os.path.join(root, f))
        print(f"Found n {len(video_files)} videos")
        return video_files

    def extract_frames(self, root_dir, size, output_dir):

        video_files = NTU_extractor._get_ntu_video_files(root_dir)

        for seq, video_file in enumerate(tqdm(video_files)):
            seq_name = os.path.basename(video_file).split(".")[0]
            seq_dir_parents = os.path.dirname(video_file).replace(root_dir, "")

            vcap_video = cv2.VideoCapture(video_file)

            seq_dir = os.path.join(output_dir, seq_dir_parents, seq_name)
            imgs_dir = os.path.join(seq_dir, 'imgs')
            os.makedirs(imgs_dir, exist_ok=True)

            if vcap_video.isOpened():
                fps = int(vcap_video.get(cv2.CAP_PROP_FPS))

                frame_count = int(vcap_video.get(cv2.CAP_PROP_FRAME_COUNT))
                print("Extracting")

                for id in tqdm(range(frame_count)):
                    # Capture frame-by-frame
                    _, frame = vcap_video.read()

                    frame = cv2.resize(frame, size)
                    cv2.imwrite(os.path.join(imgs_dir, f'frame{id:07d}.png'),
                                frame)

                with open(os.path.join(seq_dir, 'fps.txt'), 'w') as f:
                    f.write(str(fps))


class MPII_extractor:
    def _get_mpii_video_files(input_dir):
        video_files = []
        for root, dirs, files in os.walk(input_dir):
            for f in files:
                base_dir = os.path.split(root)[-1]
                if f.endswith('.avi') and base_dir == 'imageSequence':
                    video_files.append(os.path.join(root, f))
        print(f"Found n {len(video_files)} videos")
        return video_files

    def extract_frames(self, root_dir, size, output_dir):
        video_files = MPII_extractor._get_mpii_video_files(root_dir)

        for seq, video_file in enumerate(tqdm(video_files)):
            seq_name = os.path.basename(video_file).split(".")[0]
            seq_dir_parents = os.path.dirname(video_file).replace(root_dir, "")
            chair_mask_file = video_file.replace('imageSequence', 'ChairMasks')
            fg_mask_file = video_file.replace('imageSequence', 'FGmasks')

            vcap_video = cv2.VideoCapture(video_file)
            vcap_chair = cv2.VideoCapture(chair_mask_file)
            vcap_fg = cv2.VideoCapture(fg_mask_file)

            seq_dir = os.path.join(output_dir, seq_dir_parents, seq_name)
            imgs_dir = os.path.join(seq_dir, 'imgs')
            os.makedirs(imgs_dir, exist_ok=True)

            if vcap_video.isOpened() and vcap_fg.isOpened(
            ) and vcap_chair.isOpened():
                fps = int(vcap_video.get(cv2.CAP_PROP_FPS))

                frame_count = int(vcap_video.get(cv2.CAP_PROP_FRAME_COUNT))
                print("Extracting")

                for id in tqdm(range(frame_count)):
                    # Capture frame-by-frame

                    _, frame = vcap_video.read()
                    _, chair_mask = vcap_chair.read()
                    _, fg_mask = vcap_fg.read()

                    frame = apply_masks_to_frame(frame, [chair_mask, fg_mask])
                    frame = cv2.resize(frame, size)
                    cv2.imwrite(os.path.join(imgs_dir, f'frame{id:07d}.png'),
                                frame)

                with open(os.path.join(seq_dir, 'fps.txt'), 'w') as f:
                    f.write(str(fps))


def apply_masks_to_frame(video_frame, masks=None):
    result = video_frame
    for mask in masks:
        mask_to_apply = np.repeat(np.expand_dims(mask[:, :, 2], -1) > 0, 3, -1)
        result = result * mask_to_apply

    return result
