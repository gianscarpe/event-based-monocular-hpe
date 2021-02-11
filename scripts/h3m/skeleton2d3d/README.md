# Skeleton2D3D

Code for reproducing the results in the following paper:

This repo, together with [image-play](https://github.com/ywchao/image-play) and [pose-hg-train (branch `image-play`)](https://github.com/ywchao/pose-hg-train/tree/image-play), hold the code for reproducing the results in the following paper:

**Forecasting Human Dynamics from Static Images**  
Yu-Wei Chao, Jimei Yang, Brian Price, Scott Cohen, Jia Deng  
IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017  

Check out the [project site](http://www-personal.umich.edu/~ywchao/image-play/) for more details.

### Role

- The role of this repo is to implement **training step 2** (Sec. 3.3), i.e. pre-training a 3D skeleton converter to recover 3D joint locations from 2D heatmaps.

- This is later used to initialize the 3D skeleton converter sub-network in **training step 3** (Sec. 3.3), i.e. training the full system.

### Citing Skeleton2D3D

Please cite Skeleton2D3D if it helps your research:

    @INPROCEEDINGS{chao:cvpr2017,
      author = {Yu-Wei Chao and Jimei Yang and Brian Price and Scott Cohen and Jia Deng},
      booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      title = {Forecasting Human Dynamics from Static Images},
      year = {2017},
    }

### Clone the Repository

This repo contains one submodules (`pose-hg-train`), so make sure you clone with `--recursive`:

  ```Shell
  git clone --recursive https://github.com/ywchao/skeleton2d3d.git
  ```

### Contents

1. [Download Pre-Computed Models and Prediction](#download-pre-computed-models-and-prediction)
2. [Dependencies](#dependencies)
3. [Setting Up Human3.6M](#setting-up-human36m)
4. [Setting Up Penn Action](#setting-up-penn-action)
5. [Training 3D Skeleton Converter on Ground-Truth Heatmaps](#training-3d-skeleton-converter-on-ground-truth-heatmaps)
6. [Fine-Tuning 3D Skeleton Converter on Predicted Heatmaps](#fine-tuning-3d-skeleton-converter-on-predicted-heatmaps)
7. [Comparison with Zhou et al. [40]](#comparison-with-zhou-et-al-40)
8. [Evaluation](#evaluation)

## Download Pre-Computed Models and Prediction

If you just want to run the training of the full system, i.e. [image-play](https://github.com/ywchao/image-play), you can simply download the pre-computed models and prediction (108M) and skip the remaining content.

  ```Shell
  ./scripts/fetch_s2d3d_models_prediction.sh
  ./scripts/setup_symlinks_models.sh
  ```

This will populate the `exp` folder with `precomputed_s2d3d_models_prediction` and set up a set of symlinks.

You can also now [set up Human3.6M](#setting-up-human36m) and [run the evaluation demo](#evaluation) with the downloaded prediction. This will ensure exact reproduction of the paper's results.

## Dependencies

To proceed to the remaining content, make sure the following are installed.

- [Torch7](https://github.com/torch/distro)
    - We used [commit bd5e664](https://github.com/torch/distro/commit/bd5e664194953539e928546e987c615a481a8eee) (2016-10-17) with CUDA 8.0.27 RC and cuDNN v5.1 (cudnn-8.0-linux-x64-v5.1).
    - All our models were trained on a GeForce GTX TITAN X GPU.
- [matio-ffi](https://github.com/soumith/matio-ffi.torch)
- [torch-hdf5](https://github.com/deepmind/torch-hdf5)
- [MATLAB](https://www.mathworks.com/products/matlab.html)

## Setting Up Human3.6M

The Human3.6M dataset is used for training and evaluation.

1. Download the [Human3.6M dataset](https://vision.imar.ro/human3.6m). Only the Poses RawAngles and Videos files are required:

    ```Shell
    Poses_RawAngles_S1.tgz
    Poses_RawAngles_S5.tgz
    Poses_RawAngles_S6.tgz
    Poses_RawAngles_S7.tgz
    Poses_RawAngles_S8.tgz
    Poses_RawAngles_S9.tgz
    Poses_RawAngles_S11.tgz
    Videos_S1.tgz
    Videos_S5.tgz
    Videos_S6.tgz
    Videos_S7.tgz
    Videos_S8.tgz
    Videos_S9.tgz
    Videos_S11.tgz
    ```

    Place these files under `external/Human3.6M`.

2. Extract the files:

    ```Shell
    for i in external/Human3.6M/*.tgz; do tar zxvf $i -C external/Human3.6M; done
    ```

    This will populate the `external/Human3.6M` folder with `S1`, `S5`, `S6`, `S7`, `S8`, `S9`, and `S11`.

3. Download the Human3.6M dataset code:

    ```Shell
    ./h36m_utils/fetch_h36m_code.sh
    ```

    This will populate the `h36m_utils` folder with `Release-v1.1`.

4. Generate meta files. Start MATLAB `matlab` under `skeleton2d3d`. You should see the message `added paths for the experiment!` followed by the MATLAB prompt `>>`. Run the following command:

    ``` Shell
    H36MDataBase.instance;
    ```

    Set the data path to `external/Human3.6M` and the config file directory to `h36m_utils/Release-v1.1`. This will create a new file `H36M.conf` under `skeleton2d3d`.

5. Preprocess data for training and evaluation:

    ```Shell
    matlab -r "generate_data_h36m; quit"
    ```

    This will populate the `data/h36m` folder with `frames`, `train.mat`, and `val.mat`.

6. **Optional:** Visualize 3D pose sequences:


    ```Shell
    matlab -r "vis_3d_pose; quit"
    ```

    The output will be saved in `output/vis_3d_pose`.

## Setting Up Penn Action

The Penn Action dataset is used for running prediction.

1. Download the [Penn Action dataset](https://upenn.box.com/PennAction) to `external`. `external` should contain `Penn_Action.tar.gz`. Extract the files:

    ```Shell
    tar zxvf external/Penn_Action.tar.gz -C external
    ```

    This will populate the `external` folder with a folder `Penn_Action` with `frames`, `labels`, `tools`, and `README`.

2. Preprocess Penn Action by cropping the images:

    ```Shell
    matlab -r "prepare_penn_crop; quit"
    ```

    This will populate the `data/penn-crop` folder with `frames` and `labels`.

3. Generate validation set and preprocess annotations:

    ```Shell
    matlab -r "generate_valid_penn; quit"
    python tools/preprocess.py
    ```

    This will populate the `data/penn-crop` folder with `valid_ind.txt`, `train.h5`, `val.h5`, and `test.h5`.

## Training 3D Skeleton Converter on Ground-Truth Heatmaps

We begin with training a 3D skeleton converter on Human3.6M. As the first step, we use ground-truth heatmaps as input to the network.

1. Before starting, make sure to remove the symlinks from the download section, if any:

    ```Shell
    find exp -type l -delete
    ```

2. **Optional:** Visualize training examples. Each example consists of input ground-truth heamaps and ground-truth 3D pose. The heamaps are artificially generated by projecting 3D pose onto the image plane. This is done in Torch7 each time we load a training sample. We provide a way to visualize this process in MATLAB:

    ```Shell
    matlab -r "vis_pose_proj; quit"
    ```

    The output will be saved in `output/vis_pose_proj`.

3. Start training:

    ```Shell
    ./scripts/h36m/res-64.sh $GPU_ID
    ```

    The output will be saved in `exp/h36m/res-64`.

4. **Optional:** Visualize training loss and accuracy:

    ```Shell
    matlab -r "plot_loss_err; quit"
    ```

    The output will be saved to `output/plot_res-64.pdf`.

5. **Optional:** Visualize prediction on a subset of the validation set:

    ```Shell
    matlab -r "vis_preds_h36m; quit"
    ```

    The output will be saved in `output/vis_res-64/h36m_val`. The predicted pose is colored by blue, green, and red, and the ground-truth pose is colored by black, cyan, and magenta.

6. **Optional:** Run prediction on Penn Action. Given the Human3.6M trained model, we can run prediction on Penn Action. Again, we use ground-truth heatmaps as input to the network. Note that Penn Action contains unlabeled joints, which will introduce empty heatmaps that were not seen during training.

    ```Shell
    ./scripts/penn-crop/res-64-pred.sh $GPU_ID
    ```

    The output will be saved in `exp/penn-crop/res-64`.

7. **Optional:** Visualize prediction on a subset of the validation set:

    ```Shell
    matlab -r "vis_preds_penn; quit"
    ```

    The output will be saved in `output/vis_res-64/penn_val`.

## Fine-Tuning 3D Skeleton Converter on Predicted Heatmaps

Rather than ground-trtuh heatmaps, often times the 3D skeleton converter is expected to take predicted heatmaps as input. We next fine-tune the pre-trained 3D skeleton converter on heatmaps produced by an hourglass network.

1. Obtain a trained hourglass model. This is done with the submodule `pose-hg-train`. 

    **Option 1:** [Download pre-computed hourglass models (50M)](https://github.com/ywchao/pose-hg-train/tree/image-play#downloading-pre-computed-hourglass-models): **(recommended)**

    ```Shell
    cd pose-hg-train
    ./scripts/fetch_hg_models.sh
    ./scripts/setup_symlinks_models.sh
    cd ..
    ```

    This will populate the `pose-hg-train/exp` folder with `precomputed_hg_models` and set up a set of symlinks.

    **Option 2:** [Train your own models](https://github.com/ywchao/pose-hg-train/tree/image-play#training-your-own-models).

2. Start training:

    ```Shell
    ./scripts/h36m/hg-256-res-64-hg0-hgfix.sh $GPU_ID
    ````

    The output will be saved in `exp/h36m/hg-256-res-64-hg0-hgfix`.

3. **Optional:** Visualize training loss, error, and accuracy:

    ```Shell
    matlab -r "plot_loss_err_acc; quit"
    ```

    The output will be saved to `output/plot_hg-256-res-64-hg0-hgfix.pdf`.

4. **Optional:** Visualize prediction on a subset of the validation set:

    ```Shell
    matlab -r "vis_preds_h36m_hg; quit"
    ```

    The output will be saved in `output/vis_hg-256-res-64-hg0-hgfix/h36m_val`. The predicted pose is colored by blue, green, and red, and the ground-truth pose is colored by black, cyan, and magenta.

5. **Optional:** Run prediction on Penn Action. Again, rather than using ground-truth heatmaps as in the last section, we use predicted heatmaps as input here.

    ```Shell
    ./scripts/penn-crop/hg-256-res-64-hg0-hgfix-pred.sh $GPU_ID
    ```

    The output will be saved in `exp/penn-crop/hg-256-res-64-hg0-hgfix`.

6. **Optional:** Visualize prediction on a subset of the validation set:

    ```Shell
    matlab -r "vis_preds_penn_hg; quit"
    ```

    The output will be saved in `output/vis_hg-256-res-64-hg0-hgfix/penn_val`.

## Comparison with Zhou et al. [40]

This demo shows how we compare 3D pose recovery with [Zhou et al. [40]](https://fling.seas.upenn.edu/~xiaowz/dynamic/wordpress/shapeconvex/) in the paper (Sec. 4.2).

1. Fine-tune the hourglass network on Human3.6M. We will use the hourglass output as input to Zhou et al.'s method. Our goal is to evaluate the 3D pose output on Human3.6M. Since the hourglass model from `pose-hg-train` is trained on MPII and Penn Action, we first fine-tune it on Human3.6M:

    ```Shell
    ./scripts/h36m/hg-256.sh $GPU_ID
    ```

    The output will be saved in `exp/h36m/hg-256`.

2. Run prediction:

    ```Shell
    ./scripts/h36m/hg-256-pred.sh $GPU_ID
    ````

    The output will be saved in `exp/h36m/hg-256`.

3. Download Zhou et al.'s MATLAB code:

    ```Shell
    ./shapeconvex/fetch_shapeconvex.sh
    ````

    This will populate the `shapeconvex` folder with `release`.

4. Learn pose dictionary on Human3.6M:

    ```Shell
    matlab -r "shapeconvex_dl; quit"
    ```

    The output will be saved to `shapeconvex/shapeDict_h36m.mat`.

5. Run 3D pose estimation:

    ```Shell
    matlab -r "shapeconvex_run; quit"
    ```

    The output will be saved to `shapeconvex/res_hg-256-pred/h36m_val`.

6. **Optional:** Visualize prediction on a subset of the validation set:

    ```Shell
    matlab -r "shapeconvex_vis; quit"
    ```

    The output will be saved to `shapeconvex/vis_hg-256-pred/h36m_val`.

7. Finally, for a fair comparison, we also need to fine-tune our 3D skeleton converter using the fine-tuned hourglass.

    ```Shell
    ./scripts/h36m/hg-256-res-64-hg1-hgfix.sh $GPU_ID
    ```

    The output will be saved in `exp/h36m/hg-256-res-64-hg1-hgfix`.

## Evaluation

This demo runs the MATLAB evaluation script and reproduces our results in the paper (Tab. 2). If you are using [pre-computed prediction](#download-pre-computed-models-and-prediction), and want to also output Zhou et al.'s results, make sure to first run step 3, 4, and 5 in the last section.

1. Compute mean per joint position errors (MPMJE):

    ```Shell
    matlab -r "eval_run; quit"
    ```

    This will print out the MPMJE values.

2. **Optional:** Visualize Zhou et al.'s and our results.

    ```Shell
    matlab -r "vis_run; quit"
    ```

    The output will be saved in `evaluation/shapeconvex` and `evaluation/skeleton2d3d`.
