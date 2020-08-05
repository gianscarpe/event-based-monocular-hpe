import argparse
import os

import pandas as pd
from tqdm import tqdm

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _get_existing_exps(root_path, postfix_path):
    result = []
    subdirs = os.listdir(root_path)
    for s in subdirs:
        path = os.path.join(root_path, s, postfix_path)
        if os.path.exists(path):
            result.append(s)

    return result


def _extract(path, exp_name, exp_metrics, test_metric):
    event_acc = EventAccumulator(path)
    event_acc.Reload()
    exp = {}
    for t in exp_metrics:
        w_times, step_nums, vals = zip(*event_acc.Scalars(t))
        exp['steps'] = step_nums
        exp[t] = vals

    df_exp = pd.DataFrame(exp)

    w_times, step_nums, vals = zip(*event_acc.Scalars(test_metric))
    df_summary = pd.DataFrame([[exp_name, vals[-1]]],
                              columns=['exp', test_metric])
    return df_summary, df_exp


def get_pd_collection(root_path, postfix_path, exp_metrics, test_metric):
    summary = pd.DataFrame([], columns=['exp', test_metric])
    exps = {}
    exp_names = _get_existing_exps(root_path, postfix_path)

    for exp_name in tqdm(exp_names):
        path = os.path.join(root_path, exp_name, postfix_path)
        try:
            df_summary, df_exp = _extract(path, exp_name, exp_metrics,
                                          test_metric)
            summary = summary.append(df_summary)
            exps[exp_name] = df_exp
        except Exception as ex:
            print(f"Error with {path}")
            print(ex)
            continue

    return summary, exps


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Extract tensoboard meaningfull information')

    parser.add_argument(
        '--root_path',
        required=False,
        default='/home/gianscarpe/dev/exps/voxelgrid_pose/exps_resnet34/',
        help='Base exps path')

    parser.add_argument('--output_dir',
                        required=False,
                        default='.',
                        help='Output base dir')

    parser.add_argument('--exp_metrics',
                        nargs='+',
                        required=False,
                        default=['val_loss'])

    parser.add_argument('--test_metric', required=False, default=['val_loss'])

    args = parser.parse_args()
    root_path = args.root_path
    out_dir = args.output_dir
    exp_metrics = args.exp_metrics
    test_metric = args.test_metric
    default_postfix_path = 'tb_logs/default/version_0/'

    summary, exps = get_pd_collection(root_path, default_postfix_path,
                                      exp_metrics, test_metric)

    summary.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    for exp_name, exp_df in exps.items():
        exp_df.to_csv(os.path.join(out_dir, f'{exp_name}.csv'), index=False)
