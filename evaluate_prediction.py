from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys
import numpy as np
from scipy.signal import argrelextrema

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_prediction(gold_summary, predictions):
    f1 = exact_match = total = 0.
    exact_match += metric_max_over_ground_truths(
        exact_match_score, prediction, ground_truths)
    f1 += metric_max_over_ground_truths(
        f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

def evaluate_preds_two_hot(gold_file_summaries, gold_file_indices, predictions, train_name):
    correct_preds, total_correct, total_preds, number_indices = 0., 0., 0., 0.
    start_num_exact_correct, end_num_exact_correct = 0, 0
    gold_standard_summaries = open(gold_file_summaries, 'r')
    gold_indices = open(gold_file_indices, 'r')
    file_name = train_name + "/" + str(time.time()) + ".txt"
   
    with open(file_name, 'a') as f:
        for batch_preds in self.output(sess):
            for preds in batch_preds:
                index_prediction = preds
                gold = gold_indices.readline()
                gold = gold.split()
                gold_start = int(gold[0])
                gold_end = int(gold[1])

                np_preds = np.asarray(index_prediction.tolist())
                maxima = argrelextrema(np_preds, np.greater)[0] 
                tuples = [(x, np_preds[x]) for x in maxima]
                maxima = sorted(tuples, key = lambda x: x[1])
                start_index = min(maxima[-1], maxima[-2])[0]
                end_index = max(maxima[-1], maxima[-2])[0]

                print(gold_start)
                print(start_index)
                print(gold_end)
                print (end_index)

                text = file_dev.readline()
                summary = ' '.join(text.split()[start_index:end_index])
                gold_summary = ' '.join(text.split()[gold_start:gold_end])
                summary = normalize_answer(summary)
                gold_summary = normalize_answer(gold_summary)

                f.write(summary + ' \n')
                f.write(gold_summary + ' \n')

                x = range(start_index,end_index + 1)
                y = range(gold_start,gold_end + 1)
                xs = set(x)
                overlap = xs.intersection(y)
                overlap = len(overlap)

                if start_index == gold_start:
                    start_num_exact_correct += 1
                if end_index == gold_end:
                    end_num_exact_correct += 1
                
                number_indices += 1
                correct_preds += overlap
                total_preds += len(x)
                total_correct += len(y)

        start_exact_match = start_num_exact_correct/number_indices
        end_exact_match = end_num_exact_correct/number_indices
        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

        gold_standard.close()

        f.write('Model results: \n')
        f.write('learning rate: %d \n' % self.lr)
        f.write('batch size: %d \n' % self.batch_size)
        f.write('hidden size: %d \n' % self.hidden_size)
        f.write('bill_length: %d \n' % self.bill_length)
        f.write('bill_file: %s \n' % self.train_data_file)
        f.write('dev_file: %s \n' % self.dev_data_file)
        f.write("Epoch start_exact_match/end_exact_match/P/R/F1: %.2f/%.2f/%.2f/%.2f/%.2f \n" % (start_exact_match, end_exact_match, p, r, f1))
        f.close()
    
    return (start_exact_match, end_exact_match), (p, r, f1)

def main():
    a = np.array([1,2,3,4,5,4,3,2,1,2,3,2,1,2,3,4,5,6,5,4,3,2,1])
    maxima = argrelextrema(a, np.greater)[0] 
    tuples = [(x, a[x]) for x in maxima]
    maxima = sorted(tuples, key = lambda x: x[1])
    index_start = min(maxima[-1], maxima[-2])[0]
    index_end = max(maxima[-1], maxima[-2])[0]
    print(index_start)
    print(index_end)

if __name__ == "__main__":
    main()

