# Event-camera

* Train classification models based on ResNet18, Resnet34, ...
* Train 3D reconstruction models (In progress)
* Dataset adpatation for DHP19 dataset
* Generate events from video dataset with different frames representations
  (constant-count, spatiotemporal voxelgrid)

## Table of contents
- [DHP19 tools](#dhp19-tools)
- [Environment](#environment)
- [Train and evaluate classifier](#train-and-evaluate-classifier)
- [Events from video](#events-from-video)

## DHP19 tools
It's provided a toolset for generating `.npy` event frames from `DHP19`
dataset. The supported representation are: `spatiotemporal voxelgrid` and
`constant count`. Adapt `rootCodeFolder`, `rootDataFolder` and
`outDatasetFolder` to your setup. More at
[https://sites.google.com/view/dhp19/home][DHP19 official page].

```
matlab -r "run('./tools/generate_DHP19/Generate_DHP19.m')"
```

## Environment
The following packages are required: `torch >=1.4` `cv2` `opencv` `tqdm`
`scikit-video` `eigen` `boost` `boost-cpp` `pybind11` `albumentations` `hydra` `pytorch_lightning`. 
Create a `conda` environment using provided `environment.yml`. `cuda_version`
must be adapt to your current setup.

```
cuda_version=10.1

conda env create -f ./environment.yml
conda install -y pytorch>=1.4 torchvision cudatoolkit=$cuda_version -c pytorch
conda activate event-camera
```

In order to use `generate.py` you must build `esim_py`. (Please
check if you are using `event-camera` conda environment)
```
git clone https://github.com/uzh-rpg/rpg_vid2e
cd esim_py
python -m pip install .
```

## Train and evaluate classifier
If you want to begin experiming with the classifier:
```
python train.py 
```
A complete configuration is provided at `./confs/train/config.yaml`.  In
particular, refer to `./confs/train/dataset/...` for dataset configuration
(including `path` specification). Compatibility is guarantee with
`time_constant` and `voxel` representations. If you want to continue an ended
experiment, you can set `training.load_training` to `true` and provide a
checkpoint path:
```
python train.py training.load_training=true training.load_path={YOUR_MODEL_CHECKPONT}
```

To evaluate a model, you can use:
```
python eveluate.py training.load_path={YOUR_MODEL_CHECKPOINT}
```

## Events from video
It's provided a tool for generate events frames from standard videos. Standard
configuration is provided at `./confs/generate/config.yaml`. Currently there're
two supported representations: `voxel` and `constant_count`. 

Launch using:
```
conda activate event-camera
python generate.py input_path={YOUR_INPUT_PATH} output_path={YOUR_OUTPUT_PATH} representation={voxel|time_constant}
```
Please refer to [rpg_vid2e](https://github.com/uzh-rpg/rpg_vid2e) for more information about simulator parameters.




