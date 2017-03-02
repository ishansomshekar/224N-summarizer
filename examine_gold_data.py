import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from os import getcwd
from os import listdir
from collections import Counter
import statistics
import pandas
import csv

data_datapath = getcwd() +'/ALL_GOLD_SUMMARIES/'

def return_files(path):
    return [path+f for f in listdir(path) if (not f.startswith('.') and not f.startswith('missing'))]

def return_dir(path):
    return [path+f + '/' for f in listdir(path) if (not f.startswith('.'))]


standard_file = "bills_with_gold.csv"
bills_with_gold = set()
names_already_seen = set()

def main():
    #create set with names of files with gold summaries for easy look-up
    with open(standard_file, 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile.read().splitlines())]
        for row in rows:
            name = row[0]
            bills_with_gold.add(name)

    #get file lengths
    dataset_len = 0
    file_count = 0
    file_directories = return_dir(data_datapath)
    text_file_lens = []
    bill_length_and_names = []
    dupes = []
    for dir_path in file_directories:
        print "reading directory: ", dir_path
        dataset_len += len(listdir(dir_path))
        file_names = return_files(dir_path) #([dir_path + '/' + name for name in listdir(dir_path)])
        for file in file_names:
            #make sure that this is a summary that has a corresponding summary

            index_start = file.rfind('/') + 1
            index_end = file.rfind('_')
            file_name = file[index_start:index_end]
            
            if file_name in bills_with_gold:
                if file_name not in names_already_seen:
                    names_already_seen.add(file_name)
                    file_text = open(file, "r")
                    wordcount = len(file_text.read().split())
                    text_file_lens.append(wordcount)
                    bill_length_and_names.append([wordcount, file])
                    file_count += 1
                    if file_count % 1000 == 0:
                        print "finished ", file_count
    # print text_file_lens
    # print
    # print dataset_len

    # text_file_lens = sorted(text_file_lens)
    # print text_file_lens
    # print
    median = statistics.median(text_file_lens)
    print "median ", median
    print

    with open('ONLY_GOLD_SUMMARIES_WITH_BILLS.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for x in bill_length_and_names:
            writer.writerow(x)

    print("Finished splitting dataset!")

if __name__ == "__main__":
    main()