import numpy as np
from os import getcwd
from os import listdir
import csv


def return_files(path):
    return [path+f for f in listdir(path) if (not f.startswith('missing_files') and not f.startswith('.'))]

def return_dir(path):
    return [path+f + '/' for f in listdir(path) if (not f.startswith('.'))]


def main():
    datapaths = [getcwd() + '/ALL_CLEAN_BILLS/']
    standard_file = "bills_with_gold.csv"
    bills_with_gold = set()
    names_already_seen = set()
    #create set with names of files with gold summaries for easy look-up
    path_to_file_name_dict = dict()
    path_to_file_name_dict[getcwd() +'/ALL_CLEAN_BILLS/'] = "bills_info.csv"
    path_to_file_name_dict[ getcwd() +'/ALL_GOLD_SUMMARIES/'] = "summaries_info.csv"

    with open(standard_file, 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile.read().splitlines())]
        for row in rows:
            name = row[0]
            bills_with_gold.add(name)

    for path in datapaths:
        dataset_len = 0
        file_count = 0
        print "analyzing path: ", path
        file_directories = return_dir(path)
        bill_length_and_names = []
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
                        bill_length_and_names.append([file_name, file, wordcount])
                        file_count += 1
                        if file_count % 1000 == 0:
                            print "finished ", file_count

        ## MAKE SURE TO UPDATE THE FILE NAME
        with open(path_to_file_name_dict[path], 'wb') as csvfile:
            writer = csv.writer(csvfile)
            for x in bill_length_and_names:
                writer.writerow(x)

    print("Finished analyzing dataset!")

if __name__ == "__main__":
    main()