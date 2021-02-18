# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import sys
import zipfile
from glob import glob
from shutil import rmtree

import h5py
import numpy as np

sys.path.append("../")

output_filename = "data_3d_h36m"
output_filename_2d = "data_2d_h36m_gt"
subjects = ["S1", "S5", "S6", "S7", "S8", "S9", "S11"]

if __name__ == "__main__":
    if os.path.basename(os.getcwd()) != "data":
        print('This script must be launched from the "data" directory')
        exit(0)

    parser = argparse.ArgumentParser(
        description="Human3.6M dataset downloader/converter"
    )

    # Convert dataset preprocessed by Martinez et al. in https://github.com/una-dinosauria/3d-pose-baseline
    parser.add_argument(
        "--from-archive",
        default="",
        type=str,
        metavar="PATH",
        help="convert preprocessed dataset",
    )

    # Convert dataset from original source, using files converted to .mat (the Human3.6M dataset path must be specified manually)
    # This option requires MATLAB to convert files using the provided script
    parser.add_argument(
        "--from-source",
        default="",
        type=str,
        metavar="PATH",
        help="convert original dataset",
    )

    # Convert dataset from original source, using original .cdf files (the Human3.6M dataset path must be specified manually)
    # This option does not require MATLAB, but the Python library cdflib must be installed
    parser.add_argument(
        "--from-source-cdf",
        default="",
        type=str,
        metavar="PATH",
        help="convert original dataset",
    )

    args = parser.parse_args()

    if args.from_archive and args.from_source:
        print("Please specify only one argument")
        exit(0)

    if os.path.exists(output_filename + ".npz"):
        print("The dataset already exists at", output_filename + ".npz")
        exit(0)

    if args.from_source:
        print("Converting original Human3.6M dataset from", args.from_source)
        output = {}

        from scipy.io import loadmat

        for subject in subjects:
            output[subject] = {}
            file_list = glob(
                # Full instead of D3_positions
                args.from_source
                + "/"
                + subject
                + "/MyPoseFeatures/D3_positions/*.mat"
            )
            assert len(file_list) == 30, (
                "Expected 30 files for subject "
                + subject
                + ", got "
                + str(len(file_list))
            )
            for f in file_list:
                action = os.path.splitext(os.path.splitext(os.path.basename(f))[0])[0]

                if subject == "S11" and action == "Directions":
                    continue  # Discard corrupted video

                # Use consistent naming convention
                canonical_name = action.replace("TakingPhoto", "Photo").replace(
                    "WalkingDog", "WalkDog"
                )

                hf = loadmat(f)
                if "F" in hf:
                    data = hf["F"]
                elif "data" in hf:
                    data = hf["data"]
                positions = data[0, 0].reshape(-1, 32, 3)
                positions /= 1000  # Meters instead of millimeters
                output[subject][canonical_name] = positions.astype("float32")

        print("Saving...")
        np.savez_compressed(output_filename, positions_3d=output)

        print("Done.")

    elif args.from_source_cdf:
        print(
            "Converting original Human3.6M dataset from",
            args.from_source_cdf,
            "(CDF files)",
        )
        output = {}

        import cdflib

        for subject in subjects:
            output[subject] = {}
            file_list = glob(
                args.from_source_cdf
                + "/"
                + subject
                + "/MyPoseFeatures/D3_Positions/*.cdf"
            )
            assert len(file_list) == 30, (
                "Expected 30 files for subject "
                + subject
                + ", got "
                + str(len(file_list))
            )
            for f in file_list:
                action = os.path.splitext(os.path.basename(f))[0]

                if subject == "S11" and action == "Directions":
                    continue  # Discard corrupted video

                # Use consistent naming convention
                canonical_name = action.replace("TakingPhoto", "Photo").replace(
                    "WalkingDog", "WalkDog"
                )

                hf = cdflib.CDF(f)
                positions = hf["Pose"].reshape(-1, 32, 3)
                positions /= 1000  # Meters instead of millimeters
                output[subject][canonical_name] = positions.astype("float32")

        print("Saving...")
        np.savez_compressed(output_filename, positions_3d=output)

        print("Done.")

    else:
        print("Please specify the dataset source")
        exit(0)
