import argparse
import csv
import os
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi eval script.')
    parser.add_argument('--summary_path', type=str)
    args = parser.parse_args()
    csv_path = args.summary_path

    with open(csv_path) as csvfile:
        experiments = csv.DictReader(csvfile)        
        for exp in experiments:
            if not os.path.exists(
                    os.path.join(exp['load_path'], 'results.json')):
                command = f"python evaluate_dhp19.py training=margipose load_path={exp['load_path']}"
                print(command)
                #           subprocess.call(command, shell=True)
