import cv2
import os
import subprocess
import torch
import shutil
import esim_py
import numpy as np
import os
from matplotlib import pyplot as plt
import glob
import sys

def extract_frames(video_files, size, output_dir):
    for seq, video_file in enumerate(video_files):
        seq_name = os.path.basename(video_file).split(".")[0]
        vcap = cv2.VideoCapture(video_file)
        seq_dir = os.path.join(output_dir, seq_name)
        imgs_dir = os.path.join(seq_dir, 'imgs')
        os.makedirs(imgs_dir, exist_ok=True)

        if vcap.isOpened():
            print(seq_name)
            W  = int(vcap.get(3)) # float
            H = int(vcap.get(4)) # float

            print('width, height:', W, H)

            fps = int(vcap.get(cv2.CAP_PROP_FPS))
            print('fps:', fps)  # float
            print('timestamp between frames:', 1/fps)
            #print(cv2.CAP_PROP_FPS) # 5

            frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
            print('frames count:', frame_count)  # float
            for id in range(frame_count):
                # Capture frame-by-frame
                _, frame = vcap.read()
                frame = cv2.resize(frame, size)
                cv2.imwrite(os.path.join(imgs_dir, f'frame{id}.png'), frame)
            with open(os.path.join(seq_dir, 'fps.txt'), 'w') as f:
                f.write(str(fps))


    
    

