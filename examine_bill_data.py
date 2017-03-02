import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from os import getcwd
from os import listdir
from collections import Counter
import statistics
import pandas
import csv

data_datapath = getcwd() +'/ALL_CLEAN_BILLS/'

def return_files(path):
    return [path+f for f in listdir(path) if (isfile(join(path, f)) and not f.startswith('.'))]

def return_dir(path):
    return [path+f for f in listdir(path) if (not f.startswith('.'))]

def main():

    #get file lengths
    dataset_len = 0
    file_count = 0
    file_directories = return_dir(data_datapath)
    text_file_lens = []
    bill_length_and_names = []
    for dir_path in file_directories:
        print "reading directory: ", dir_path
        dataset_len += len(listdir(dir_path))
        file_names = ([dir_path + '/' + name for name in listdir(dir_path)])
        for file in file_names:
            file_text = open(file, "r")
            wordcount = len(file_text.read().split())
            text_file_lens.append(wordcount)
            bill_length_and_names.append([wordcount, file])
            file_count += 1
            if file_count % 1000 == 0:
                print "finished ", file_count
    assert len(text_file_lens) == dataset_len

    text_file_lens = sorted(text_file_lens)
    print text_file_lens
    print
    median = statistics.median(text_file_lens)
    print "median ", median

    with open('BILL_LENGTHS.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for x in bill_length_and_names:
            writer.writerow(x)

    # file_size_counts = Counter(text_file_lens)
    # the histogram of the data
    # weights = [50 for x in range(0, 150)]
    # weights += [10 for x in range(0, 151)]
    # print len(weights)
    # n, bins, patches = plt.hist(text_file_lens, facecolor='green', bins=[x * 1000 for x in range(0,300)], weights = [])

    # plt.xlabel('file_size')
    # plt.ylabel('probability')
    # plt.title('Text lengths')
    # plt.axis([0, 320000, 0,100000])
    # plt.grid(True)

    # plt.show()

    # enc_train, enc_eval, enc_test = split_dataset()
    # write_encset(enc_train, enc_eval, enc_test)
    print("Finished splitting dataset!")

if __name__ == "__main__":
    main()