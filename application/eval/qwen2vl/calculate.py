import os
import argparse
import re

from collections import defaultdict
import json


def calculate_metrics(labels, predictions, answers, scores_file, bad_cases_file):
    pos = 1
    neg = 0
    TP, TN, FP, FN = 0, 0, 0, 0
    fns = []
    fps = []

    for i, (pred, label) in enumerate(zip(predictions, labels)):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
            if len(answers):
                fps.append(answers[i])
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1
            if len(answers):
                fns.append(answers[i])  # Store the full answer object

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    yes_ratio = predictions.count(1) / len(predictions)

    scores_file.write('TP\t\tFP\t\tTN\t\tFN\t\t\n')
    scores_file.write('{}\t\t{}\t\t{}\t\t{}\n'.format(TP, FP, TN, FN))
    scores_file.write(f'Accuracy: {acc:.3f}\n')
    scores_file.write(f'Precision: {precision:.3f}\n')
    scores_file.write(f'Recall: {recall:.3f}\n')
    scores_file.write(f'F1 score: {f1:.3f}\n')
    scores_file.write(f'Yes ratio: {yes_ratio:.3f}\n')
    scores_file.write('%.3f, %.3f, %.3f, %.3f, %.3f\n\n' % (f1, acc, precision, recall, yes_ratio))

    bad_cases_file.write("\nFalse Positives:\n")
    for answer in fps:
        bad_cases_file.write(f"Answer: {answer}\n")

    bad_cases_file.write("\nFalse Negatives:\n")
    for answer in fns:
        pass
        bad_cases_file.write(f"Answer: {answer}\n")
    return f1, acc, precision, recall, yes_ratio


def calculate(category, results_dir):
    answers = [json.loads(q) for q in open(os.path.join(str(answers_dir), f"{category}_answers.jsonl"))]
    labels = [json.loads(line) for line in open(os.path.join(annotations_dir, file), 'r')]
    grouped_answers = defaultdict(list)
    label_dict = {label['question_id']: {'label': 'yes' if (label['label'] == '是的' or label['label'] == 'yes') else 'no', 'category': label['category'], 'topic': label['topic']} for label in labels}

    for answer in answers:
        text = answer['text']
        if text.find('.') != -1 or text.find('。') != -1 or text.find('，') != -1:
            text = text.split('.')[0]
            text = text.split(',')[0]
            text = text.split('。')[0]
            text = text.split('，')[0]
        text = text.replace(',', '').replace('。', '')
        words = text.split(' ')
        if 'No' in words or 'no' in words or '不是' in words:
            answer['text'] = 'no'
        if 'Yes' in words or 'yes' in words or '是的' in words:
            answer['text'] = 'yes'
        words = ''.join(words)
        if words and bool(re.fullmatch(r"^(A?B?C?D?)$", words)):
            answer['text'] = ''.join(words)
        question_id = answer['question_id']
        if question_id in label_dict:
            label_info = label_dict[question_id]
            grouped_answers[label_info['topic']].append((answer['text'], label_info['label'], answer))
    scores_file = open(os.path.join(results_dir, f"{category}_scores.txt"), "w")
    bad_cases_file = open(os.path.join(results_dir, f"{category}_badcases.txt"), "w")
    overall_labels = []
    overall_predictions = []
    scores_file.write('Category: {}, # samples: {}\n'.format(category, len(answers)))
    for topic, pairs in grouped_answers.items():
        scores_file.write(f'Category: {category}, Topic: {topic}\n')
        bad_cases_file.write(f'Category: {category}, Topic: {topic}\n')
        predictions = [1 if p[0] == 'yes' else 0 for p in pairs]
        labels = [1 if p[1] == 'yes' else 0 for p in pairs]
        answers = [p[2] for p in pairs]
        calculate_metrics(labels, predictions, answers, scores_file, bad_cases_file)
        overall_labels.extend(labels)
        overall_predictions.extend(predictions)
    scores_file.write(f"Category Overall Result: {category}\n")
    calculate_metrics(overall_labels, overall_predictions, [], scores_file, bad_cases_file)

    scores_file.close()
    bad_cases_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=str)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--category", type=str, default="all")
    args = parser.parse_args()
    annotations_dir = os.path.join(args.eval_dir, "annotations")
    answers_dir = os.path.join(args.eval_dir, "answers", args.model_name)
    results_dir = os.path.join(args.eval_dir, "results", args.model_name)
    os.makedirs(results_dir, exist_ok=True)
    for file in os.listdir(annotations_dir):
        category = file[:-18]
        if args.category not in ['all', category]:
            continue
        calculate(category, results_dir)

