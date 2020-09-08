import argparse
import csv
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi eval script.')
    parser.add_argument('--summary_path', type=str)
    args = parser.parse_args()
    csv_path = args.summary_path

    with open(csv_path) as csvfile:
        experiments = csv.DictReader(csvfile)
        for exp in experiments:
            if not os.path.exists(
                    os.path.join(exp['load_path'], 'auc.json')):
                command = f"python evaluate_dhp19.py training=margipose gpus='0'        load_path={exp['load_path']}".replace(
                    "$", "\\$")
                print(command)
                os.system(command)
