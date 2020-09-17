import argparse
import csv
import json
import os
import pickle

import numpy as np


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


def get_latex_from_numpy(table, headings, caption=""):
    result = "\n \n \\begin{table}\n \n \\caption{" + caption + "}"
    result += "\\centering \n \\resizebox{\\columnwidth}{!}{% \n " + "\\begin{tabular}{l" + "r" * (
        table.shape[-1]) + "|r}\n \\hline \\hline \n "

    for head in headings:
        result += "& " + head
    result += "& mean movement"
    result += "\\\\ \\hline \\hline \n"

    for i in range(0, 33):
        result += f"movement {i+1}"
        values_per_exp = table[i, :]
        for v in values_per_exp:
            result += f"& {v:.2f}"

        # MEAN COLUMN
        result += f"& {values_per_exp.mean():.2f}"
        result += " \\\\ \n \n"

    # MEAN ROW
    result += "\\hline \\hline mean"
    for col in range(table.shape[-1]):
        result += f"& {table[:, col].mean():.2f} "

    result += f"& {table.mean():.2f}"
    result += "\\end{tabular}}\\end{table}"
    result = result.replace('_', '\\_')

    return result


def load_results(full_exps_results, metric):
    table_columns = []
    table_headings = []
    if metric in full_exps_result:
        item = full_exps_result[metric]
        table_columns = item['table_columns']
        table_headings = item['table_headings']

    return table_columns, table_headings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract tensoboard meaningfull information')

    parser.add_argument('--summary_path', type=str, help='Base exps path')
    parser.add_argument('--dump_path',
                        type=str,
                        default='./dump.pickle',
                        help='Base exps path')
    parser.add_argument('--metrics',
                        type=str,
                        nargs="+",
                        default=['test_meanMPJPE', 'test_meanPCK'],
                        help='Base exps path')

    args = parser.parse_args()
    summary_path = args.summary_path
    metrics = args.metrics
    loaded_results = {}
    full_exps_result = {}

    if os.path.exists(args.dump_path):
        with open(args.dump_path, 'rb') as dump_read:
            loaded_results = pickle.load(dump_read)

    with open(summary_path) as csvfile:
        experiments = list([*csv.DictReader(csvfile)])

        with open("table.tex", "w") as tf:
            tf.write("")
        for metric in metrics:
            table_columns, table_headings = load_results(
                loaded_results, metric)

            for ind, exp in enumerate(experiments):
                json_file = os.path.join(exp['load_path'], 'result.json')
                if not os.path.exists(json_file):
                    print(f"Error with {json_file}")
                    continue

                with open(json_file) as js:
                    exp_name = get_exp_acron(exp)
                    if exp_name in table_headings:
                        continue
                    table_headings.append(exp_name)
                    print(exp_name)
                    results = json.load(js)
                    values = list(results[metric].values())
                    table_columns.append(values)

            sorted_index = np.argsort(table_headings)
            table_columns = np.array(table_columns)[sorted_index]
            table = np.stack(table_columns, -1)
            table_headings = list(sorted(table_headings))

            full_exps_result[metric] = {
                'table_columns': table_columns,
                'table_headings': table_headings
            }

            result = get_latex_from_numpy(table, table_headings, metric)
            with open("table.tex", "a") as tf:
                tf.write(result)

    with open("dump.pickle", "wb") as dump_file:
        pickle.dump(full_exps_result, dump_file)
