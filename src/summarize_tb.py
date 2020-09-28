import argparse
import os
from pathlib import Path  # if you haven't already done so

import pandas as pd
import tabulate
from tqdm import tqdm

from experimenting.agents import MargiposeEstimator
from experimenting.utils import get_checkpoint_path


def get_exp_acron(exp):
    result = ""
    if exp['training']['backbone'] == "none":
        if exp['training']['pretrained']:
            result += "imagenet_"
        else:
            result += "raw_"
    else:
        result += "pretrained_"

    result += exp['training']['model']
    return result


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
    params = model._hparams
    print("model loaded")
    exp = {
        'exp_name': get_exp_acron(params),
        'load_path': path,
        'checkpoint': load_path
    }
    for param in exp_params:
        params_tree = params
        for p in param.split('/'):
            params_tree = params_tree[p]
        exp[param] = params_tree
    print("Parameter extracted")
    return exp


def get_pd_collection(root_path, postfix_path, exp_metrics, results_files):
    exps = []
    exp_names = _get_existing_exps(root_path, postfix_path)
    names = []
    for exp_name in tqdm(exp_names):
        exp_path = os.path.join(root_path, exp_name)
        path = os.path.join(exp_path, postfix_path)
        try:
            exp = _extract(path, exp_name, exp_metrics)
            if exp['exp_name'] in names:
                print(f"Exp {exp} is a duplicate")
            else:
                names.append(exp['exp_name'])
            for r in results_files:
                exp[r] = os.path.exists(os.path.join(path, r))
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

    parser.add_argument('--reval', action='store_true')

    parser.add_argument(
        '--exp_params',
        nargs='+',
        required=False,
        default=['training/backbone', 'training/pretrained', 'training/model'])

    parser.add_argument('--results',
                        nargs='+',
                        default=['torso.json', 'basic.json'],
                        help='Results')

    args = parser.parse_args()
    root_path = args.root_path
    results_files = args.results
    out_dir = args.root_path
    exp_params = args.exp_params
    reval = args.reval
    default_postfix_path = 'checkpoints'

    if reval:
        exps_pd = pd.DataFrame(
            get_pd_collection(root_path, default_postfix_path, exp_params,
                              results_files))
        exps_pd.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    else:
        exps_pd = pd.read_csv(os.path.join(out_dir, 'summary.csv'))

    print(tabulate.tabulate(exps_pd, headers='keys'))
