import numpy as np
import ast
from scipy.signal import argrelextrema
from os import listdir
# from evaluate_prediction import normalize_answer
import os

#gold_summaries_file = "summaries_dev_bills_4_150.txt"
gold_indices_file = "indices_dev_bills_4_150.txt"

pred_folder = 'ptr_adam_100'

def return_files(path):
    return [path+ "/" +f for f in listdir(path) if f.startswith('preds_')]

def localminima(file):
    #gold_summaries = open(gold_summaries_file, 'r')
    correct_preds, total_correct, total_preds, number_indices = 0., 0., 0., 0.
    start_num_exact_correct, end_num_exact_correct = 0, 0
    gold_indices = open(gold_indices_file, 'r')
    lengths = []
    with open(file) as f:
        while True:
            first = f.readline()
            if not first:
                break
            a = ast.literal_eval(first)
            a = np.asarray(a)
            b = ast.literal_eval(f.readline())
            b = np.asarray(b)
            f.readline()

            a = np.exp(a - np.amax(a))
            a = a / np.sum(a)
            b = np.exp(b - np.amax(b))
            b = b / np.sum(b)

            start_maxima = argrelextrema(a, np.greater)[0]
            tuples = [(x, a[x]) for x in start_maxima]
            start_maxima = sorted(tuples, key = lambda x: x[1])
            if len(start_maxima) > 0:
                a_idx = start_maxima[-1][0]
            else:
                a_idx = np.argmax(a)

            end_maxima = argrelextrema(b, np.greater)[0]
            tuples = [(x, b[x]) for x in end_maxima if x > a_idx]
            end_maxima = sorted(tuples, key = lambda x: x[1])
            if len(end_maxima) > 0:
                b_idx = end_maxima[-1][0]
            else:
                b_idx = np.argmax(b)

            gold = gold_indices.readline()
            gold = gold.split()
            gold_start = int(gold[0])
            gold_end = int(gold[1])
            start_index = int(a_idx)
            end_index = int(b_idx)
            lengths.append(end_index - start_index)

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

    print "local minima:   "
    print start_exact_match, end_exact_match, p, r, f1
    print "mean: "
    mean = sum(lengths)/len(lengths)
    print sum(lengths)/len(lengths)
    return start_exact_match, end_exact_match, p, r, f1, mean


def neither_fixed(file):
    correct_preds, total_correct, total_preds, number_indices = 0., 0., 0., 0.
    start_num_exact_correct, end_num_exact_correct = 0, 0
    gold_indices = open(gold_indices_file, 'r')
    lengths = []
    with open(file) as f:
        while True:
            first = f.readline()
            if not first:
                break
            a = ast.literal_eval(first)
            a = np.asarray(a)
            b = ast.literal_eval(f.readline())
            b = np.asarray(b)
            f.readline()

            a = np.exp(a - np.amax(a))
            a = a / np.sum(a)
            b = np.exp(b - np.amax(b))
            b = b / np.sum(b)

            a_idx = len(a) - 2
            b_idx = len(b) - 1

            b_max = b_idx
            total_max = a[a_idx] * b[b_max]

            for i in xrange(len(a)-3, -1, -1):
                if b[i + 1] > b[b_max]:
                    b_max = i + 1
                if a[i] * b[b_max] > total_max:
                    a_idx = i
                    b_idx = b_max

            gold = gold_indices.readline()
            gold = gold.split()
            gold_start = int(gold[0])
            gold_end = int(gold[1])
            start_index = int(a_idx)
            end_index = int(b_idx)
            lengths.append(end_index - start_index)

            x = range(start_index,end_index + 1)
            y = range(gold_start,gold_end + 1)
            xs = set(x)
            overlap = xs.intersection(y)
            overlap = len(overlap)
            # print(start_index, end_index)
            # print (gold_start, gold_end)
            # print
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

    print "neither_fixed:"
    print start_exact_match, end_exact_match, p, r, f1
    print "mean: "
    print sum(lengths)/len(lengths)
    mean = sum(lengths)/len(lengths)
    return start_exact_match, end_exact_match, p, r, f1, mean

def end_max_fixed(file):
    correct_preds, total_correct, total_preds, number_indices = 0., 0., 0., 0.
    start_num_exact_correct, end_num_exact_correct = 0, 0
    gold_indices = open(gold_indices_file, 'r')
    lengths = []
    with open(file) as f:
        while True:
            first = f.readline()
            if not first:
                break
            if first[:3] == 'end':
                break
            a = ast.literal_eval(first)
            # print a
            a = np.asarray(a)

            # ep = f.readline()
            b = ast.literal_eval(f.readline())
            # print b
            b = np.asarray(b)

            f.readline()

            # print a
            a = np.exp(a - np.amax(a))
            a = a / np.sum(a)
            b = np.exp(b - np.amax(b))
            b = b / np.sum(b)

            b_idx = np.argmax(b)
            #if out of bounds, fix it
            a_idx = b_idx

            count = 0
            while a_idx >= b_idx and count != 0:
                a_idx = np.argmax(a)
                a[a_idx] = 0
                count += 1

            lengths.append(b_idx - a_idx)

            gold = gold_indices.readline()
            gold = gold.split()
            # print gold
            gold_start = int(gold[0])
            gold_end = int(gold[1])
            start_index = int(a_idx)
            end_index = int(b_idx)

            x = range(start_index,end_index + 1)
            y = range(gold_start,gold_end + 1)
            xs = set(x)
            overlap = xs.intersection(y)
            overlap = len(overlap)
            # print(start_index, end_index)
            # print (gold_start, gold_end)
            # print
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

    print "end_max_fixed:"
    print start_exact_match, end_exact_match, p, r, f1
    print "mean: "
    print sum(lengths)/len(lengths)

    return start_exact_match, end_exact_match, p, r, f1, mean

def max_fixed(file):
    correct_preds, total_correct, total_preds, number_indices = 0., 0., 0., 0.
    start_num_exact_correct, end_num_exact_correct = 0, 0
    gold_indices = open(gold_indices_file, 'r')
    lengths = []
    with open(file) as f:
        while True:
            first = f.readline()
            if not first:
                break
            a = ast.literal_eval(first)
            # print a
            a = np.asarray(a)

            # ep = f.readline()
            b = ast.literal_eval(f.readline())
            # print b
            b = np.asarray(b)

            f.readline()

            # print a
            a = np.exp(a - np.amax(a))
            a = a / np.sum(a)
            b = np.exp(b - np.amax(b))
            b = b / np.sum(b)

            a_idx = np.argmax(a)
            #if out of bounds, fix it
            b_idx = a_idx
            count = 0
            while b_idx <= a_idx and count != 5:
                b_idx = b.index(max(b))
                b[b_idx] = 0
                count += 0

            lengths.append(b_idx - a_idx)

            gold = gold_indices.readline()
            gold = gold.split()
            # print gold
            gold_start = int(gold[0])
            gold_end = int(gold[1])
            start_index = int(a_idx)
            end_index = int(b_idx)

            x = range(start_index,end_index + 1)
            y = range(gold_start,gold_end + 1)
            xs = set(x)
            overlap = xs.intersection(y)
            overlap = len(overlap)
            # print(start_index, end_index)
            # print (gold_start, gold_end)
            # print
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

    print "start_max_fixed:"
    print start_exact_match, end_exact_match, p, r, f1
    print "mean: "
    print sum(lengths)/len(lengths)

    return start_exact_match, end_exact_match, p, r, f1, mean

maxf1_neither = 0
maxFile_neither = None
evaluated = None
sem = None
eem = None
p_fin = None
r_fin = None
mean_fin = None
files = return_files(os.getcwd() +'/' + pred_folder)
pred_analysis = open(pred_folder + "pred_analysis.txt", "w")

maxFile_local = None
maxf1_local = 0
for file in files:
    print file
    pred_analysis.write(file + "\n")
    #start_exact_match, end_exact_match, p, r, f1, mean = max_fixed(file)
    #start_exact_match, end_exact_match, p, r, f1, mean = end_max_fixed(file)
    pred_analysis.write("local max\n")
    start_exact_match, end_exact_match, p, r, f1, mean = localminima(file)
    pred_analysis.write(str(start_exact_match) + " " + str(end_exact_match) + " " + str(p) + " " + str(r) + " " + str(f1) + " " + str(mean) + "\n")
    if f1 > maxf1_local:
        maxf1_local = f1
        maxFile_local = file
        # evaluated = 'local min'
        # sem = start_exact_match
        # eem = end_exact_match
        # p_fin = p
        # r_fin = r
        # mean_fin = mean
    pred_analysis.write("neither fixed\n")
    start_exact_match, end_exact_match, p, r, f1, mean = neither_fixed(file)
    pred_analysis.write(str(start_exact_match) + " " + str(end_exact_match) + " " + str(p) + " " + str(r) + " " + str(f1) + " " + str(mean) + "\n")
    if f1 > maxf1_neither:
        maxf1_neither = f1
        maxFile_neither = file
        # evaluated = 'neither fixed'
        # sem = start_exact_match
        # eem = end_exact_match
        # p_fin = p
        # r_fin = r
        # mean_fin = mean
    pred_analysis.write("\n")

print "THIS IS THE MAX NEITHER"
pred_analysis.write("THIS IS THE MAX NEITHER\n")
print maxFile_neither
pred_analysis.write(maxFile_neither + "\n")
neither_fixed(maxFile_neither)

pred_analysis.write("THIS IS THE MAX LOCAL\n")
print maxFile_local
pred_analysis.write(maxFile_local + "\n")
localminima(maxFile_local)

# print "final scores:"
# print maxFile
# print sem, eem, p_fin, r_fin, mean_fin, maxf1, evaluated

