import argparse
import csv
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Multi eval script.')
    parser.add_argument('--summary_path', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--protocol', default='cross-view', type=str)
    parser.add_argument('--batch_size', default='32', type=str)
    parser.add_argument('--metrics',
                        nargs="+",
                        default=['AUC', 'PCK', 'MPJPE'],
                        type=str)
    parser.add_argument('--gpus', default='1', type=str)
    parser.add_argument('--result_file', default='result.json')
    parser.add_argument('--estimate_depth', default='false')

    args = parser.parse_args()
    gpus = args.gpus
    csv_path = args.summary_path
    dataset = args.dataset
    batch_size = args.batch_size    
    metrics = "[" + ",".join(args.metrics) + "]"
    result_file = args.result_file
    protocol = args.protocol
    estimate_depth = args.estimate_depth

    done = []
    with open(csv_path) as csvfile:
        experiments = csv.DictReader(csvfile)
        for exp in experiments:
            result_path = os.path.join(exp['load_path'], result_file)
            if not os.path.exists(result_path):
                command = f"python evaluate_dhp19.py training.batch_size={batch_size} training.metrics={metrics} result_file={result_file} training.estimate_depth={estimate_depth} training=margipose dataset.partition={protocol} dataset={dataset} gpus=[{gpus}] load_path={exp['load_path']}".replace(
                    "$", "\\$")
                print(command)
                os.system(command)
