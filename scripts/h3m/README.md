
## First steps

0. Request a LICENSE from academic porpoise for Human3.6m 
1. Download and extract human3.6m dataset. You can use the tool inside `download_human3.6m` or download them by hand from http://vision.imar.ro/human3.6m/description.php. NOTE: either case, you need to request a LICENSE from the authors

## Joints 
In order to generate good-quality labels, you need to use full joints positions (`D3_Positions`). These data are not distributed in human3.6m, which instead gives access to `RAW_Angles` only. We provide a matlab script to convert `Raw_angles` into `Full_D3_Positions` 

0. Get a valid `MATLAB` installation. We tested the script on MATLAB2021a
1. Launch Matlab into `Generate_Full_D3_Positions`
2. Run script `generate.m`. It asks to specify the original directories of the dataset. After the process, you should have a `D3_Full_Positions` in  your dataset

## Process

2. Use `event_library` generator script to generate `raw` events from `mp4` files:
```
python event_library/tools/generate.py frames_dir=path/to/dataset out_dir=out upsample=true emulate=true search=false representation=raw
```
3. Launch `pre.py` to generate `constant_count` frames and joints
