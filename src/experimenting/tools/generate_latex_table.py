import argparse
import json
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract tensoboard meaningfull information')
    result = "\\begin{table}\n \caption{} \n \centering \n \\begin{tabular}{lr}\n \hline \n "
    parser.add_argument('--json_files', nargs='+', help='Base exps path')
    args = parser.parse_args()
    json_files = args.json_files
    values = {}
    for json_file in json_files:
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
        result += " \\\\ \n \hline \n"
    result += "mean"
    for _, v in values.items():
        v = list(v)
        result += f"& {sum(v)/len(v):.2f} "
    result += "\\end{tabular}\\end{table}"
    print(result)
    with open("table.tex", "w") as tf:
        tf.write(result)
