import numpy as np
from os import getcwd
from os import listdir
import csv


#create a dictionary that maps filename with the bills and summaries
bill_file = "ONLY_BILLS_WITH_GOLD.csv"
summary_file = "ONLY_GOLD_SUMMARIES_WITH_BILLS.csv"

name_to_summary = dict()
name_to_bill = dict()

#can preprocess in this function
def file_generator(batch_size):
    count = 0
    with open(bill_file, 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile.read().splitlines())]
        for row in rows:
            file = row[1]
            count += 1
            if count %1000 == 0:
                print "finished ", count
            index_start = file.rfind('/') + 1
            index_end = file.rfind('_')
            file_name = file[index_start:index_end]
            #second value is the length
            name_to_bill[file_name] = (file, row[0])

    with open(summary_file, 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile.read().splitlines())]
        for row in rows:
            file = row[1]
            count += 1
            if count %1000 == 0:
                print "finished ", count
            index_start = file.rfind('/') + 1
            index_end = file.rfind('_')
            file_name = file[index_start:index_end]
            #second value is the length
            name_to_summary[file_name] = (file, row[0])

    current_batch_summaries = []
    current_batch_bills = []
    with open("dummy_dataset.csv", 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile.read().splitlines())]
        for row in rows:
            file_name = row[0]
            
            bill_text = open(name_to_bill[file_name][0], 'r').read().split()
            current_batch_bills.append(bill_text)
            
            summary_text = open(name_to_summary[file_name][0], 'r').read().split()
            current_batch_summaries.append(summary_text)
            
            if len(current_batch_summaries) == batch_size:
                yield current_batch_bills, current_batch_summaries
                current_batch_bills = []
                current_batch_summaries = []
   