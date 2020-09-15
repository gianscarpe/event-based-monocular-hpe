import argparse
import csv
import json
import os


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
    parser = argparse.ArgumentParser(
        description='Extract tensoboard meaningfull information')

    parser.add_argument('--summary_path', type=str, help='Base exps path')
    parser.add_argument('--metrics',
                        type=str,
                        nargs="+",
                        default=['test_meanMPJPE', 'test_meanPCK'],
                        help='Base exps path')
    args = parser.parse_args()
    summary_path = args.summary_path
    metrics = args.metrics

    with open(summary_path) as csvfile:
        experiments = list([*csv.DictReader(csvfile)])
        with open("table.tex", "w") as tf:
            tf.write("")
        for metric in metrics:
            values = {}
            result = "\n \n \\begin{table}\n \n \\caption{" + metric + "} \\centering \n \\resizebox{\\columnwidth}{!}{% \n " + "\\begin{tabular}{l" + "r" * (
                len(experiments)) + "|r}\n \\hline \\hline \n "

            for ind, exp in enumerate(experiments):
                json_file = os.path.join(exp['load_path'], 'result.json')
                if not os.path.exists(json_file):
                    print(f"Error with {json_file}")
                    continue

                with open(json_file) as js:
                    exp_name = get_exp_acron(exp)
                    print(exp_name)
                    print(json_file)
                    results = json.load(js)

                    values[exp_name] = results[metric].values()
                    if len(results[metric].values()) < 33:
                        print("ERROR!")

            for k, v in values.items():
                result += "& " + k
            result += "& mean movement"
            result += "\\\\ \\hline \\hline \n"

            values_items = values.items()
            for i in range(0, 33):
                result += f"movement {i+1}"
                values_per_movement = []
                for _, v in values_items:
                    v = list(v)
                    result += f"& {v[i]:.2f}"
                    values_per_movement.append(v[i])
                result += f"& {sum(values_per_movement) / len(values_per_movement):.2f}"
                result += " \\\\ \n \n"
            result += "\\hline \\hline mean"
            for _, v in values.items():
                v = list(v)
                result += f"& {sum(v)/len(v):.2f} "

            result += f"& {sum([sum(v) for _, v in values.items()]) /            sum([len(v) for _, v in values.items()]):.2f}"
            result += "\\end{tabular}}\\end{table}"
            result = result.replace('_', '\\_')
            with open("table.tex", "a") as tf:
                tf.write(result)
