[![Build Status](https://travis-ci.com/gianscarpe/event-based-monocular-hpe.svg?branch=master)](https://travis-ci.com/gianscarpe/event-based-monocular-hpe)
[![Documentation
Status](https://readthedocs.org/projects/event-camera/badge/?version=latest)](https://event-camera.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/gianscarpe/event-based-monocular-hpe/badge.svg?branch=master)](https://coveralls.io/github/gianscarpe/event-based-monocular-hpe?branch=master)
# Event-camera

* Train classification models based on ResNet18, Resnet34, ...
* Train 3D reconstruction models
* Dataset adpatation for DHP19 dataset
* Generate events from events dataset with different frames representations
  (constant-count, spatiotemporal voxelgrid)

## Table of contents
- [Environment](#environment)
- [Data](#data)
- [Model zoo](#model-zoo)
- [Agents](#agents)


## Environment
Create a `virtualenv` environment from `requirements.txt`. 
Using pipenv:

```
pipenv install -r requirements.txt
pipenv shell

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
python evaluate_dhp19.py training={TASK} dataset={DATASET_REPRESENTATION} dataset.partition={cross-view|cross-subject} load_path={YOUR_MODEL_CHECKPOINT}
```

Experimenting implementations for classification, autoencoder, 2d joints estimation,
heatmap prediction, and 3d joints estimation are based on  `pytorch_lighting` framework.
