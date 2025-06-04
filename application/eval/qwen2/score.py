# -*- coding: utf-8 -*-
# @project: LLaMA-Factory
# @filename: predict.py
# @author: Karl Wu
# @contact: wlt1990@outlook.com
# @time: 2025/6/3 15:01
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def calculate_metrics(prediction_data):

    if not prediction_data:
        raise ValueError("Prediction data is empty")

    total_samples = len(prediction_data)
    beam_size = len(prediction_data[0]['output']) if total_samples > 0 else 0

    # Initialize counters for all possible k values up to beam_size
    max_k = min(10, beam_size)
    k_values = sorted({1, 2, 3, 5, 10, beam_size}.intersection(range(1, beam_size + 1)))

    topk_acc_counts = {k: 0 for k in k_values}
    topk_valid_counts = {k: 0 for k in k_values}
    kth_acc_counts = {k: 0 for k in k_values}
    kth_valid_counts = {k: 0 for k in k_values}

    # Process each sample
    for sample in tqdm(prediction_data, desc="Processing predictions"):
        outputs = sample['output']

        # Calculate top-k and k-th metrics
        for k in k_values:
            top_k_outputs = outputs[:k]

            # Top-k accuracy (any correct in top k)
            if any(output['is_correct'] for output in top_k_outputs):
                topk_acc_counts[k] += 1

            # K-th accuracy (only the k-th prediction)
            if k <= len(outputs) and outputs[k - 1]['is_correct']:
                kth_acc_counts[k] += 1

            # Top-k validity (sum of valid in top k)
            topk_valid_counts[k] += sum(1 for output in top_k_outputs if output['is_valid'])

            # K-th validity (only the k-th prediction)
            if k <= len(outputs):
                kth_valid_counts[k] += 1 if outputs[k - 1]['is_valid'] else 0

    # Calculate rates
    def calculate_rate(count_dict, denominator):
        return {k: count / denominator[k] for k, count in count_dict.items()}

    accuracy_rates = calculate_rate(topk_acc_counts, {k: total_samples for k in k_values})
    kth_accuracy_rates = calculate_rate(kth_acc_counts, {k: total_samples for k in k_values})

    validity_rates = calculate_rate(topk_valid_counts, {k: total_samples * k for k in k_values})
    kth_validity_rates = calculate_rate(kth_valid_counts, {k: total_samples for k in k_values})

    return {
        'total_samples': total_samples,
        'beam_size': beam_size,
        'accuracy_rates': accuracy_rates,
        'kth_accuracy_rates': kth_accuracy_rates,
        'validity_rates': validity_rates,
        'kth_validity_rates': kth_validity_rates,
        'k_values': k_values
    }


def save_results(results, output_path, prediction_file_name):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(f"PREDICTION_FILE: {prediction_file_name}\n")
        f.write(f"Total samples: {results['total_samples']}\n")
        f.write(f"Beam size: {results['beam_size']}\n\n")

        # Write top-k metrics
        f.write("Top-k Accuracy:\n")
        for k in results['k_values']:
            f.write(f"Top-{k}: {results['accuracy_rates'][k]:.4f}\n")

        f.write("\nK-th Accuracy:\n")
        for k in results['k_values']:
            f.write(f"K={k}: {results['kth_accuracy_rates'][k]:.4f}\n")

        f.write("\nTop-k Validity Rate:\n")
        for k in results['k_values']:
            f.write(f"Top-{k}: {results['validity_rates'][k]:.4f}\n")

        f.write("\nK-th Validity Rate:\n")
        for k in results['k_values']:
            f.write(f"K={k}: {results['kth_validity_rates'][k]:.4f}\n")


def load_prediction_data(prediction_file_path):
    with open(prediction_file_path, 'r') as f:
        return json.load(f)


def main(args):
    prediction_file_path = Path(args.prediction_dir) / f"{args.prediction_file}.json"
    output_path = Path(args.output_dir) / f"{args.prediction_file}.txt"

    try:
        prediction_data = load_prediction_data(prediction_file_path)
        results = calculate_metrics(prediction_data)
        save_results(results, output_path, args.prediction_file)
        print(f"Metrics calculation completed successfully. Results saved to {output_path}")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate prediction metrics")
    parser.add_argument('--prediction_file', type=str, default='qwen205_moltrans_mit_separated_space_lora_para1_epoch3',
                        help="Name of the prediction file (without extension)")
    parser.add_argument('--prediction_dir', type=str,
                        default="/home/liangtao/Development/LLMSpace/LLaMA-Factory/results/prediction",
                        help="Directory containing prediction files")
    parser.add_argument('--output_dir', type=str,
                        default="/home/liangtao/Development/LLMSpace/LLaMA-Factory/results/scores",
                        help="Directory to save results")
    main(parser.parse_args())