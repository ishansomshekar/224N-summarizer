import numpy as np
from os import getcwd
from os import listdir
import tensorflow as tf
import csv

#can preprocess in this function
def file_generator(batch_size, bill_data_path, summary_data_path):    
    current_batch_summaries = []
    current_batch_bills = []
    counter = 0
    with tf.gfile.GFile('bill_data_100.txt', mode="r") as source_file:
        with tf.gfile.GFile('summary_data_100.txt', mode="r") as target_file:
            for bill in source_file:
                summary = target_file.readline()
                counter += 1
                current_batch_bills.append(bill)
                current_batch_summaries.append(summary)
                if len(current_batch_summaries) == batch_size:
                    yield current_batch_bills, current_batch_summaries
                    current_batch_bills = []
                    current_batch_summaries = []

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






