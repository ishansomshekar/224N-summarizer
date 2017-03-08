import numpy as np
from os import getcwd
from os import listdir
import tensorflow as tf
import csv

#can preprocess in this function
def file_generator(batch_size, bill_data_path, indices_data_path, sequences_data_path):    
    current_batch_summaries = []
    current_batch_bills = []
    current_batch_sequences = []
    counter = 0
    with tf.gfile.GFile(bill_data_path, mode="r") as source_file:
        with tf.gfile.GFile(indices_data_path, mode="r") as target_file:
            with tf.gfile.GFile(sequences_data_path, mode="r") as seq_file:
                for bill in source_file:
                    indices = target_file.readline()
                    sequence_len = seq_file.readline()
                    counter += 1
                    start_and_end = indices.split()
                    current_batch_bills.append(bill)
                    current_batch_summaries.append((int(start_and_end[0]), int(start_and_end[1])))
                    current_batch_sequences.append(int(sequence_len))
                    if len(current_batch_summaries) == batch_size:
                        yield current_batch_bills, current_batch_summaries, current_batch_sequences
                        current_batch_bills = []
                        current_batch_summaries = []
                        current_batch_sequences = []

    # count = 0
    # print "generating dictionaries for bills and summaries"
    # with open(bill_file, 'rb') as csvfile:
    #     rows = [row for row in csv.reader(csvfile.read().splitlines())]
    #     for row in rows:
    #         file = row[1]
    #         count += 1
    #         if count %1000 == 0:
    #             print "finished ", count
    #         index_start = file.rfind('/') + 1
    #         index_end = file.rfind('_')
    #         file_name = file[index_start:index_end]
    #         #second value is the length
    #         name_to_bill[file_name] = (file, row[0])

    # with open(summary_file, 'rb') as csvfile:
    #     rows = [row for row in csv.reader(csvfile.read().splitlines())]
    #     for row in rows:
    #         file = row[1]
    #         count += 1
    #         if count %1000 == 0:
    #             print "finished ", count
    #         index_start = file.rfind('/') + 1
    #         index_end = file.rfind('_')
    #         file_name = file[index_start:index_end]
    #         #second value is the length
    #         name_to_summary[file_name] = (file, row[0])

    # print "finished generating dictionaries for bills and summaries"
    #     current_batch_summaries = []
    # current_batch_bills = []
    # with open(bill_names_file, 'rb') as csvfile:
    #     rows = [row for row in csv.reader(csvfile.read().splitlines())]
    #     for row in rows:
    #         file_name = row[0]
            
    #         bill_text = open(name_to_bill[file_name][0], 'r').read().split()
    #         current_batch_bills.append(bill_text)
            
    #         summary_text = open(name_to_summary[file_name][0], 'r').read().split()
    #         current_batch_summaries.append(summary_text)
            
    #         if len(current_batch_summaries) == batch_size:
    #             yield current_batch_bills, current_batch_summaries
    #             current_batch_bills = []
    #             current_batch_summaries = []
    # count = 0
    # print "generating dictionaries for bills and summaries"
    # with open(bill_file, 'rb') as csvfile:
    #     rows = [row for row in csv.reader(csvfile.read().splitlines())]
    #     for row in rows:
    #         file = row[1]
    #         count += 1
    #         if count %1000 == 0:
    #             print "finished ", count
    #         index_start = file.rfind('/') + 1
    #         index_end = file.rfind('_')
    #         file_name = file[index_start:index_end]
    #         #second value is the length
    #         name_to_bill[file_name] = (file, row[0])

    # with open(summary_file, 'rb') as csvfile:
    #     rows = [row for row in csv.reader(csvfile.read().splitlines())]
    #     for row in rows:
    #         file = row[1]
    #         count += 1
    #         if count %1000 == 0:
    #             print "finished ", count
    #         index_start = file.rfind('/') + 1
    #         index_end = file.rfind('_')
    #         file_name = file[index_start:index_end]
    #         #second value is the length
    #         name_to_summary[file_name] = (file, row[0])

    # print "finished generating dictionaries for bills and summaries"
    #     current_batch_summaries = []
    # current_batch_bills = []
    # with open(bill_names_file, 'rb') as csvfile:
    #     rows = [row for row in csv.reader(csvfile.read().splitlines())]
    #     for row in rows:
    #         file_name = row[0]
            
    #         bill_text = open(name_to_bill[file_name][0], 'r').read().split()
    #         current_batch_bills.append(bill_text)
            
    #         summary_text = open(name_to_summary[file_name][0], 'r').read().split()
    #         current_batch_summaries.append(summary_text)
            
    #         if len(current_batch_summaries) == batch_size:
    #             yield current_batch_bills, current_batch_summaries
    #             current_batch_bills = []
    #             current_batch_summaries = []






