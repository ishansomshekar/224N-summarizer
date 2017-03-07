import numpy as np
import random
import codecs
import os
import shutil
import csv
from os import listdir
from os import getcwd

datafile = 'bills_with_extracted.csv'

def main():
    file_names_to_data = dict()
    count = 0
    with open(datafile, 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile.read().splitlines())]
        for row in rows:
            count +=1
            if count %1000 == 0:
                print "finished: ", count
            file_name = row[0].strip()
            file_names_to_data[file_name] = row[1:]
    dataset_length = len(file_names_to_data.items())
    print("There are %s bills" % dataset_length)
    file_names = file_names_to_data.keys()
    random.seed(12)
    dataset = np.array(random.sample(file_names, dataset_length))
    train_size = int(dataset_length*0.7)
    eval_size = int(dataset_length*0.2)
    train, evalu, test = dataset[0:train_size], dataset[train_size:train_size+eval_size], dataset[train_size+eval_size:]
    print("There are %s bills in train" % len(train))
    print("There are %s bills in dev" % len(evalu))
    print("There are %s bills in test" % len(test))

    print "creating CSV files"

    with open('train_bills.csv', 'wb') as csvfile1:
        writer1 = csv.writer(csvfile1)
        for bill_name in train:
            row = [bill_name] + file_names_to_data[bill_name]
            writer1.writerow(row)
    with open('test_bills.csv', 'wb') as csvfile2:
        writer2 = csv.writer(csvfile2)
        for bill_name in test:
            row = [bill_name] + file_names_to_data[bill_name]
            writer2.writerow(row)
    with open('dev_bills.csv', 'wb') as csvfile3:
        writer3 = csv.writer(csvfile3)
        for bill_name in evalu:
            row = [bill_name] + file_names_to_data[bill_name]
            writer3.writerow(row)

    print("Finished splitting dataset!")

if __name__ == "__main__":
    main()
