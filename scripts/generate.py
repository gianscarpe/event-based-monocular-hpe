import os

from tqdm import tqdm

import hydra
from omegaconf import DictConfig
from experimenting.generator import upsample


@hydra.main(config_path='./confs/generate/config.yaml')
def main(cfg: DictConfig):
    print(cfg.pretty())

    input_dir = cfg.input_dir
    base_output_dir = cfg.output_dir
    tmp_dir = cfg.tmp_dir
    extract = cfg.extract
    do_upsample = cfg.upsample

    tmp_frames_dir = os.path.join(tmp_dir, "frames")
    tmp_upsample_dir = os.path.join(tmp_dir, "upsample")
    extractor = hydra.utils.instantiate(cfg.extractor)
    representation = hydra.utils.instantiate(cfg.representation)
    simulator = hydra.utils.instantiate(cfg.vid2e)

    if extract:
        print("Extract RGB frames from videos")
        extractor.extract_frames(input_dir, representation.get_size(),
                                 tmp_frames_dir)

        print("Extraction completed")

    if do_upsample:
        print("Upsampling")
        upsample(tmp_frames_dir, tmp_upsample_dir)
    print("Instantiate simulator")

    video_dirs = []
    for root, dirs, _ in os.walk(tmp_upsample_dir):
        for d in dirs:
            if d == 'imgs':
                video_dirs.append(root)

    with_errors = []
    for input_video_dir in tqdm(video_dirs):
        video_struct = os.path.relpath(input_video_dir, tmp_upsample_dir)
        output_dir = os.path.join(base_output_dir, video_struct)
        try:
            simulator.generate(input_video_dir, output_dir, representation)
        except:
            print(f"Error with {input_video_dir} ")
            with_errors.append(input_video_dir)

    print(with_errors)
    print("Simulation end")
    with open(os.path.join(tmp_dir, 'errors.txt'), "w") as f:
        for item in with_errors:
            f.write("%s\n" % item)


if __name__ == '__main__':
    main()
