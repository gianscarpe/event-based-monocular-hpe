import argparse
import csv
import json
import os

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt


def get_exp_acron(exp):
    result = ""
    if exp['training/backbone'] == "none":
        if exp['training/pretrained'] == 'True':
            result += "imagenet_"
        else:
            result += "raw_"
    else:
        result += "pretrained_"

    result += exp['training/model']
    return result


if __name__ == '__main__':
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(
        description='Extract tensoboard meaningfull information')

    parser.add_argument('--summary_path', type=str, help='Base exps path')
    parser.add_argument('--metric',
                        type=str,
                        default='test_meanAUC',
                        help='Base exps path')
    args = parser.parse_args()
    summary_path = args.summary_path
    metric = args.metric
    means_per_experiment = {}
    breakpoint()
    with open(summary_path) as csvfile:
        experiments = list([*csv.DictReader(csvfile)])
        for ind, exp in enumerate(experiments):
            means = np.zeros(50)
            json_file = os.path.join(exp['load_path'], 'auc.json')
            if not os.path.exists(json_file):
                print(f"Error with {json_file}")
                continue

            with open(json_file) as js:
                exp_name = get_exp_acron(exp)
                print(exp_name)
                results = json.load(js)[metric]
                for i in range(0, 33):
                    key = f'movement_{i}'
                    means += results[key]
                means = means / 33.
                means_per_experiment[get_exp_acron(exp)] = means

        fpr = torch.linspace(0, 800, 50)
        breakpoint()
        for key in sorted(means_per_experiment.keys()):
            plt.plot(fpr, means_per_experiment[key], label=key)
        plt.xlabel("Threshold (in mm)")
        plt.ylabel("PCK")
        plt.legend()
        plt.savefig("in-deg-dist.png")
        plt.close()
