import numpy as np
from os import getcwd
from os import listdir
import csv


def main():
    standard_file = '85_PERCENTILE_BILLS.csv'

    with open('bill_names_input_85.csv', 'wb') as csvfile1:
        bill_name_writer = csv.writer(csvfile1)
        with open(standard_file, 'rb') as csvfile2:
            rows = [row for row in csv.reader(csvfile2.read().splitlines())]
            for row in rows:
                address = row[1]
                index_start = address.rfind('/') + 1
                index_end = address.rfind('_')
                file_name = address[index_start:index_end]
                bill_name_writer.writerow([file_name])

    print("Finished splitting dataset!")

if __name__ == "__main__":
    main()