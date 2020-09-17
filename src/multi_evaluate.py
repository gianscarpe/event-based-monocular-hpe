import argparse
import csv
import os

from experimenting.tools.generate_latex_table import get_exp_acron
from experimenting.utils import get_result_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi eval script.')
    parser.add_argument('--summary_path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--metrics', nargs="+", default=['AUC'], type=str)
    parser.add_argument('--result_file', default='aucs.json')
    args = parser.parse_args()
    csv_path = args.summary_path
    dataset = args.dataset
    metrics = "[" + ",".join(args.metrics) + "]"
    result_file = args.result_file
    done = []
    with open(csv_path) as csvfile:
        experiments = csv.DictReader(csvfile)
        for exp in experiments:
            exp_name = get_exp_acron(exp)
            if exp_name in done:
                continue
            
            result_path = os.path.join(exp['load_path'], result_file)
            if not os.path.exists(os.path.join(exp['load_path'], 'aucs.json')):
                command = f"python evaluate_dhp19.py training.metrics={metrics} training=margipose dataset={dataset} gpus=1 load_path={exp['load_path']}".replace(
                    "$", "\\$")
                print(command)
                os.system(command)
                done.append(exp_name)
