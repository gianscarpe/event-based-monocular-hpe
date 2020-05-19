from esim_py import EventSimulator
import os
import numpy as np

class SimulatorWrapper(EventSimulator):
    def __init__(self, Cp, Cn, refractory_period, log_eps, use_log):
        super(SimulatorWrapper, self).__init__(Cp, Cn,
                                               refractory_period,
                                               log_eps,
                                               use_log)
        

    def generate(self, input_dir, output_dir, representation):
        for seq in os.listdir(input_dir):
            seq_path = os.path.join(input_dir, seq)
            imgs_path = os.path.join(seq_path, 'imgs')
            ts_path = os.path.join(seq_path, 'timestamps.txt')
            events_output_path = os.path.join(output_dir, seq)
            os.makedirs(events_output_path, exist_ok = True)
            events = self.generateFromFolder(imgs_path, ts_path)
            
            # representation yields frame of events
            for ind, frame in enumerate(representation.frame_generator(events)):
                np.save(os.path.join(events_output_path, f"frame{ind}.npy"), frame)

        
    
