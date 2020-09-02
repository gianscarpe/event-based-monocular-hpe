import argparse
import csv
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract tensoboard meaningfull information')

    parser.add_argument('--summary_path', nargs='+', help='Base exps path')
    args = parser.parse_args()
    summary_path = args.summary_path
    with open(summary_path) as csvfile:
        experiments = csv.DictReader(csvfile)
        values = {}

        result = "\\begin{table}\n \\caption{} \n \\centering \n \\resizebox{\\columnwidth}{!}{% \n " +\ "\\begin{tabular}{l" + "r" * len(
            experiments) + "}\n \\hline \n "

        for exp in experiments:
            json_file = os.path.join(exp['load_path'], 'results.json')
            with open(json_file) as js:
                exp_name = json_file.split('/')[-3]
                results = json.load(js)
                values[exp_name] = results.values()
        for k, v in values.items():
            print(f"{k}: {v}")
            result += "& " + k
        result += "\\\\ \n"
        for i in range(0, 33):
            result += f"movement {i+1}"
            for _, v in values.items():
                v = list(v)
                result += f"& {v[i]:.2f}"
            result += " \\\\ \n \\hline \n"
        result += "mean"
        for _, v in values.items():
            v = list(v)
            result += f"& {sum(v)/len(v):.2f} "
        result += "\\end{tabular}}\\end{table}"
        result = result.replace('_', '\\_')
        with open("table.tex", "w") as tf:
            tf.write(result)
