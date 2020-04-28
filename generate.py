import os
import hydra
from omegaconf import DictConfig
from experimenting.utils import get_file_paths
import experimenting.generator as generator

@hydra.main(config_path='./confs/generate/config.yaml')
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    show_debug_images = cfg.show_debug_images
    input_dir = cfg.input_dir
    output_dir = cfg.output_dir
    
    tmp_dir = os.path.join(input_dir, 'tmp')
    tmp_frames_dir = os.path.join(tmp_dir, "frames")
    tmp_upsample_dir =  os.path.join(tmp_dir, "upsample")
    
    video_files = get_file_paths(input_dir, ['.mp4'])
    print("Extract RGB frames from videos")
    representation = hydra.utils.instantiate(cfg.representation)
    
    generator.extract_frames(video_files, representation.get_size(), tmp_frames_dir)
    #generator.upsample(tmp_frames_dir, tmp_upsample_dir)
    print("Extraction completed")

    print("Instantiate simulator")
    simulator = hydra.utils.instantiate(cfg.vid2e)
    simulator.generate(tmp_upsample_dir, output_dir, representation)
    print("Simulation end")
    
    
if __name__ == '__main__':
    main()    
