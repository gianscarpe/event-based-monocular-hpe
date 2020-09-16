import argparse
import csv
import os

from experimenting.utils import get_result_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi eval script.')
    parser.add_argument('--summary_path', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()
    csv_path = args.summary_path
    dataset = args.dataset
    
    with open(csv_path) as csvfile:
        experiments = csv.DictReader(csvfile)
        for exp in experiments:
            if not os.path.exists(os.path.join(exp['load_path'], 'aucs.json')):
                command = f"python evaluate_dhp19.py training=margipose dataset={dataset} gpus=1 load_path={exp['load_path']}".replace(
                    "$", "\\$")
                print(command)
                os.system(command)
