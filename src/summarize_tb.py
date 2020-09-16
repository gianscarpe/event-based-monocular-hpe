import argparse
import os
import shutil
from pathlib import Path  # if you haven't already done so

import pandas as pd
from tqdm import tqdm

from experimenting.agents import MargiposeEstimator
from experimenting.utils import get_checkpoint_path


def _get_existing_exps(root_path, postfix_path):
    result = []
    subdirs = os.listdir(root_path)
    for s in subdirs:
        path = os.path.join(root_path, s, postfix_path)
        if os.path.exists(path):
            result.append(s)

    return result


def _extract(path, exp_name, exp_params):
    load_path = get_checkpoint_path(path)
    model = MargiposeEstimator.load_from_checkpoint(load_path)
    print("model loaded")
    exp = {'exp_name': exp_name, 'load_path': path}
    for param in exp_params:
        params = model._hparams
        for p in param.split('/'):
            params = params[p]
        exp[param] = params
    print("Parameter extracted")
    return exp


def get_pd_collection(root_path, postfix_path, exp_metrics):
    exps = []
    exp_names = _get_existing_exps(root_path, postfix_path)

    for exp_name in tqdm(exp_names):
        exp_path = os.path.join(root_path, exp_name)
        path = os.path.join(exp_path, postfix_path)
        try:
            exp = _extract(path, exp_name, exp_metrics)
            exps.append(exp)
        except Exception as ex:
            print(f"Error with {path}")
            exp_path = Path(exp_path)
            error_path = os.path.join(exp_path.parent, 'with_errors', exp_name)
            #shutil.move(exp_path, error_path)
            print(ex)
            continue

    return exps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract tensoboard meaningfull information')

    parser.add_argument(
        '--root_path',
        required=False,
        default='/home/gianscarpe/dev/exps/voxelgrid/exps_MargiposeEstimator',
        help='Base exps path')

    parser.add_argument(
        '--exp_params',
        nargs='+',
        required=False,
        default=['training/backbone', 'training/pretrained', 'training/model'])

    args = parser.parse_args()
    root_path = args.root_path
    out_dir = args.root_path
    exp_params = args.exp_params
    default_postfix_path = 'checkpoints'

    exps_pd = pd.DataFrame(
        get_pd_collection(root_path, default_postfix_path, exp_params))
    exps_pd.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
