import os
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

import experimenting.generator as generator
from experimenting.utils import get_file_paths


def _get_mpii_video_files(input_dir):
    video_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            base_dir = os.path.split(root)[-1]
            if f.endswith('.avi') and base_dir == 'imageSequence':
                video_files.append(os.path.join(root, f))

    return video_files


@hydra.main(config_path='./confs/generate/config.yaml')
def main(cfg):
    print(cfg.pretty())

    input_dir = cfg.input_dir
    base_output_dir = cfg.output_dir
    extract = cfg.extract
    upsample = cfg.upsample

    tmp_dir = os.path.join(input_dir, 'tmp')
    tmp_frames_dir = os.path.join(tmp_dir, "frames")
    tmp_upsample_dir = os.path.join(tmp_dir, "upsample")
    representation = hydra.utils.instantiate(cfg.representation)
    if extract:
        video_files = _get_mpii_video_files(input_dir)
        print(f"Found n {len(video_files)} videos")
        print("Extract RGB frames from videos")

        generator.extract_frames(input_dir, video_files,
                                 representation.get_size(), tmp_frames_dir)
        print("Extraction completed")

    if upsample:
        print("Upsampling")
        generator.upsample(tmp_frames_dir, tmp_upsample_dir)
    print("Instantiate simulator")
    video_dirs = []
    for root, dirs, _ in os.walk(tmp_upsample_dir):
        for d in dirs:
            if d == 'imgs':
                video_dirs.append(root)

    for input_video_dir in tqdm(video_dirs):
        video_struct = os.path.relpath(input_video_dir, tmp_upsample_dir)
        output_dir = os.path.join(base_output_dir, video_struct)
        simulator = hydra.utils.instantiate(cfg.vid2e)
        simulator.generate(input_video_dir, output_dir, representation)
        del simulator
    print("Simulation end")


if __name__ == '__main__':
    main()
