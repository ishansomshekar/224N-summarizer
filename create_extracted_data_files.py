import numpy as np
from os import getcwd
from os import listdir
import csv

datafile = 'bills_with_extracted.csv'

def main():
    print "generating files"
    count = 0

    bill_data = open('bill_data_extracted_full.txt', "w")
    summary_data = open('extracted_data_full.txt', "w")
    index_data = open('indices_data_full.txt', "w")
    with open(datafile, 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile.read().splitlines())]
        for row in rows:
            file_name = row[0]
            bill_text = row[1]
            summary_text = row[3]
            start_index = row[4]
            end_index = row[5]
            
            bill_text_list = bill_text.split()
            assert ' '.join(bill_text_list[int(start_index):int(end_index)]) == summary_text

            count += 1
            if count % 1000 == 0:
                print "finished ", count

            bill_text.replace("\n", "")
            summary_text.replace("\n", "")
            start_index.replace("\n", "")
            end_index.replace("\n", "")

            bill_data.write(bill_text + "\n")
            summary_data.write(summary_text + "\n")
            index_data.write(start_index + " " + end_index + "\n")
        
        bill_data.close()
        summary_data.close()
        index_data.close()

    print("Finished analyzing dataset!")

if __name__ == "__main__":
    main()