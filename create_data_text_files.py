import numpy as np
from os import getcwd
from os import listdir
import csv

def main():
    base = "bills_3_150." 
    options = ["dev_", "train_", "test_"]
    count = 0
    for option in options:
        newbase = option + base
        csv_file_name = newbase + "csv"

        bill_data = open("bills_" + newbase + "txt", "w")
        summary_data = open("summaries_" + newbase + "txt", "w")
        sequences_data = open("sequences_" + newbase + "txt", "w")
        indices_data = open("indices_" + newbase + "txt", "w")
        #keywords_data = open("keywords_" + base + "txt", "w")
        file_name_data = open("filenames_" + newbase + "txt", "w")

        with open(csv_file_name, 'rb') as csvfile:
            rows = [row for row in csv.reader(csvfile.read().splitlines())]
            for row in rows:
                file_name = row[0]
                bill_text = row[1]
                bill_length = row[2]
                summary_text = row[3]
                summary_len = row[6]
                index_start = row[4]
                index_end = row[5]
               
                count += 1
                if count % 1000 == 0:
                    print "finished ", count

                bill_data.write(bill_text + "\n")
                summary_data.write(summary_text + "\n")
                sequences_data.write(bill_length + "\n")
                indices_data.write(index_start + " " + index_end + "\n")
                file_name_data.write(file_name + "\n")
            
        bill_data.close()
        summary_data.close()
        sequences_data.close()
        indices_data.close()
        file_name_data.close()

    print("Finished analyzing dataset!")

if __name__ == "__main__":
    main()