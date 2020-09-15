# Event-camera

* Train classification models based on ResNet18, Resnet34, ...
* Train 3D reconstruction models (In progress)
* Dataset adpatation for DHP19 dataset
* Generate events from video dataset with different frames representations
  (constant-count, spatiotemporal voxelgrid)

## Table of contents
- [Environment](#environment)
- [Data](#data)
- [Model zoo](#model-zoo)
- [Agents](#agents)
- [Events from video](#events-from-video)


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


## Data
Follow DHP19 guide


### Download DHP19
Up-to-date instruction to download ground truth, DAVIS data, and cameras
projection matrix are avaialble at [download](https://sites.google.com/view/dhp19/download?authuser=0).

### DHP19 tools
It's provided a toolset for generating `.mat` event frames from `DHP19`
dataset. The supported representation are: `spatiotemporal voxelgrid`,
`time-surfaces`, and `constant count`. Fix `rootCodeFolder`, `rootDataFolder` and
`outDatasetFolder` to your setup. More at [dhp19](https://sites.google.com/view/dhp19/home).

```
matlab -r "run('./tools/generate_DHP19/Generate_DHP19.m')"
```

The reader can generate the 2D/3D joints ground truth with. 
```
python generate_joints --homedir /path/to/dhp19 --output_dir
```

A script for converting h5 files to npy set of frames is also privided
`generate_dataset_frames`. Nevertheless, we reccomend to generate the dataset in
`.mat` directly from matlab, as explained above.

### Model zoo
A model zoo of backbones for `constant_count` and `voxelgrid` trained both with
`cross-subject` and `cross-view` protocols is publicly accessible at .

## Agents

### Train and evaluate for different tasks
If you want to begin experiming with the classifier:

```
python train.py 
```

A complete configuration is provided at `./confs/train/config.yaml`. In
particular, refer to `./confs/train/dataset/...` for dataset configuration
(including `path` specification), and to `./confs/train/training` for different
tasks. Compatibility is guarantee with `time_constant` and `voxel` representations. If you want to continue an ended
experiment, you can set `training.load_training` to `true` and provide a
checkpoint path:

```
python train.py training.load_training=true training.load_path={YOUR_MODEL_CHECKPONT}
```

To continue a previous experiment:
```
python train.py training.load_training=true training.load_path={YOUR_MODEL_CHECKPONT}
```

To train a margipose\_estimator agent:
```
python train.py training=margipose dataset=$DATASET training.latent_size=$LATENT training.model=$MODEL training.backbone=$DATASET/$BACKBONE_TASK/${MODEL}.pt loss=multipixelwise training.batch_size=$BATCH_SIZE training.stages=$N_STAGES

```

To evaluate a model, you can use:
```
python eveluate.py training.load_path={YOUR_MODEL_CHECKPOINT}
```

### Test
You can test your models using our multi-movement evaluation script. The tool
generates a `result.json` file in the provided checkpoint path.
```
python evaluate_dhp19.py training={TASK} dataset={DATASET_REPRESENTATION} load_path={YOUR_MODEL_CHECKPOINT}
```

Experimenting implementations for classification, autoencoder, 2d joints estimation,
heatmap prediction, and 3d joints estimation are based on  `pytorch_lighting` framework.

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


