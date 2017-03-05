import numpy as np
from os import getcwd
from os import listdir
import csv

MAX_THRESHOLD = 0.6

bill_file = "bills_info.csv"
summary_file = "summaries_info.csv"

name_to_summary = dict()
name_to_bill = dict()

def main():
    print "generating dictionaries for bills and summaries"
    count = 0
    with open(bill_file, 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile.read().splitlines())]
        for row in rows:
            file_name = row[0]
            length = row[2]
            file = row[1]
            count += 1
            # if count % 1000 == 0:
            #     print "finished ", count
            
            name_to_bill[file_name] = (file, length)

    with open(summary_file, 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile.read().splitlines())]
        for row in rows:
            file_name = row[0]
            length = row[2]
            file = row[1]
            count += 1
            # if count % 1000 == 0:
            #     print "finished ", count
            
            name_to_summary[file_name] = (file, length)

    print "now reading dictionaries: "
    file_count = 0
    data_within_bounds = 0
    with open('data_2x_greater_than_summaries.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for file_name, bill_adr_len in name_to_bill.items():
            summ_adr_len = name_to_summary[file_name]
            summary_len = int(summ_adr_len[1])
            bill_len = int(bill_adr_len[1])
            bill_adr = bill_adr_len[0]
            summ_adr = summ_adr_len[0]
            bill = open(bill_adr, 'r')
            bill_file_text = bill.read()
            summ_file = open(summ_adr, 'r')
            summ_file_text = summ_file.read()
            bill.close()
            summ_file.close()
            # print bill_file_text
            # print
            # print summ_file_text            
            find_index = bill_file_text.find(summ_file_text)

            split_summ = summ_file_text.split()
            for i in xrange(0, len(split_summ)-8):
                val = bill_file_text.find(' '.join(split_summ[i:i+8]))
                # print ' '.join(split_summ[i:i+8])
                if val > 0:
                    find_index = val

            # print find_index

            #open the bill and summary files from bill_adr and summ_adr
            #FIND THE SUMMARY INSIDE THE BILL 
            #you can even do... trim the bill to 400 words, and then find the summary 
            #try printing out all the different start and end indices to make sure that this is still an interesting problem
            #i'd say... if we can get ~70,000

            file_count += 1
            if file_count % 1000 == 0:
                print "finished ", file_count

            threshold_bill_len = MAX_THRESHOLD * bill_len    
            if threshold_bill_len > summary_len and summary_len < 500 and bill_len < 1300 and bill_len > 0 and summary_len > 4 and find_index > 0:
                data_within_bounds += 1
                #we have approx 112,000 bills at this point. Now, run statistics to understand the median and such
                row = [file_name, bill_adr_len[0], bill_len, summ_adr_len[0], summary_len]        
                writer.writerow(row)

    print data_within_bounds

    print("Finished analyzing dataset!")

if __name__ == "__main__":
    main()