import argparse
import csv
import json
import os
import pickle

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
    parser.add_argument('--result_file', type=str, help='Base exps path')
    parser.add_argument('--dump_path', type=str, default='./roi.pickle',  help='Base exps path')
    parser.add_argument('--metric',
                        type=str,
                        default='test_meanAUC',
                        help='Base exps path')
    args = parser.parse_args()
    summary_path = args.summary_path
    dump_path = args.dump_path
    result_file = args.result_file
    means_per_experiment = {}

    if os.path.exists(dump_path):
        with open(dump_path, 'rb') as dump_read:
            means_per_experiment = pickle.load(dump_read)
    metric = args.metric

    with open(summary_path) as csvfile:
        experiments = list([*csv.DictReader(csvfile)])
        for ind, exp in enumerate(experiments):
            means = np.zeros(30)
            json_file = os.path.join(exp['load_path'], result_file)
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

                if exp_name in means_per_experiment:
                    continue
                means_per_experiment[exp_name] = means

        fpr = torch.linspace(0, 500, 30)
        with open(dump_path, 'wb') as dump_write:
            pickle.dump(means_per_experiment, dump_write)

        for key in sorted(means_per_experiment.keys()):
            plt.plot(fpr, means_per_experiment[key], label=key)
        plt.xlabel("Threshold (in mm)")
        plt.ylabel("PCK")
        plt.legend()
        plt.savefig("roi.png")
        plt.close()

        zoom = slice(6, 15) # from 100mm to 240mm
        breakpoint()
        for key in sorted(means_per_experiment.keys()):
            plt.plot(fpr[zoom], means_per_experiment[key][zoom], label=key)
        plt.xlabel("Threshold (in mm)")
        plt.ylabel("PCK")
        plt.legend()
        plt.savefig("zoom.png")
        plt.close()        
