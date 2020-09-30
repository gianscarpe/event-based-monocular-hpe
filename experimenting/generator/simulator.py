from esim_py import EventSimulator
import os
import numpy as np
from tqdm import tqdm
import glob


class SimulatorWrapper(EventSimulator):
    def __init__(self,
                 Cp,
                 Cn,
                 refractory_period,
                 log_eps,
                 use_log,
                 batch_size=2000):
        super(SimulatorWrapper, self).__init__(Cp, Cn, refractory_period,
                                               log_eps, use_log)
        self.batch_size = batch_size

    def _get_ts_from_file(self, input_dir):
        ts_path = os.path.join(input_dir, 'timestamps.txt')
        ts = []
        with open(ts_path) as ts_file:
            for x in ts_file:
                ts.append(float(x))
        return ts

    def generate(self, input_dir, output_dir, representation):
        imgs_path = os.path.join(input_dir, 'imgs', '*')
        ts = self._get_ts_from_file(input_dir)

        images_path = sorted(glob.glob(imgs_path))
        n_images = len(images_path)
        print(f"Start generating events for {input_dir}; n {n_images}")
        os.makedirs(output_dir, exist_ok=True)
        for i in tqdm(range(n_images // self.batch_size + 1)):
            out_part_dir = os.path.join(output_dir, f"part_{i}")
            os.makedirs(out_part_dir, exist_ok=True)
            end_id = min(i + self.batch_size, n_images)
            images_input_paths = images_path[i:end_id]
            ts_input_paths = ts[i:end_id]
            events = self.generateFromStampedImageSequence(
                images_input_paths, ts_input_paths)
            # representation yields frame of events
            for ind, frame in enumerate(
                    representation.frame_generator(events)):
                np.save(os.path.join(out_part_dir, f"frame{ind:07d}.npy"),
                        frame)
            del events
