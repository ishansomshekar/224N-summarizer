import numpy as np
from os import getcwd
from os import listdir
import csv
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import random 
# bill_file = "bills_uncleaned_info.csv"
# summary_file = "summaries_uncleaned_info.csv"


name_to_summary = dict()
name_to_bill = dict()
no_summaries = set()
already_seen = set()

def main():
    THRESHOLD = 7
    THRESHOLD_WINDOW = 2
    total_count = 0
    file_count = 0
    MIN_SUMMARY_LENGTH = 5
    dupes = 0

    csv_name = 'bills_cleaned_pt2.csv'
    with open(csv_name, 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile.read().splitlines())]
        for row in rows:
            file_name = row[0]
            bill_length = int(row[2])
            bill_adr = row[1]
            name_to_bill[file_name] = (bill_adr, bill_length)

            summary_adr = row[3]
            summary_len = int(row[4])
            name_to_summary[file_name] = (summary_adr, summary_len)

    with open('bills_with_new_extracted_3.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for file_name, bill_adr_len in name_to_bill.items():
            bill_adr = bill_adr_len[0]
            bill = open(bill_adr, 'r')
            bill_file_text = bill.read()
            untouched_bill_text = bill_file_text.split()

            summary_adr_len = name_to_summary[file_name]
            summ_adr = summary_adr_len[0]
            summary = open(summ_adr, 'r')
            summary_file_text = summary.read()
            summary_len = len(summary_file_text)

            summary_file_list = summary_file_text.split()
            bill_file_list = bill_file_text.split()
            

            bill_len = len(bill_file_list)
            bill_indices = [0] * bill_len

            for word in summary_file_list:
                indices = [i for i, x in enumerate(bill_file_list) if x == word]
                for i in indices:
                    bill_indices[i] = 1
            
            ranges = []
            current_continuous_seq = []
            exists_subsequence = False
            window_counter = 0
            previous_seq = []
            for idx, val in enumerate(bill_indices):
                if val == 0: 
                    if exists_subsequence:
                        exists_subsequence = False
                        previous_seq = current_continuous_seq
                        current_continuous_seq = []
                        window_counter = 1
                    else:
                        window_counter += 1
                else:
                    if exists_subsequence:
                        current_continuous_seq.append(idx)
                    else:
                        exists_subsequence = True
                        if window_counter < THRESHOLD_WINDOW:
                            current_continuous_seq = previous_seq
                            previous_seq = []
                            current_continuous_seq.append(idx)
                            window_counter = 0
                        else:
                            current_continuous_seq.append(idx)
                            if len(previous_seq) != 0:  
                                ranges.append(previous_seq)
                            previous_seq = []
                            window_counter = 0

            sorted_ranges = sorted(ranges, key = lambda x: (len(x), max(x)), reverse = True)

            if len(sorted_ranges) > 0:
                seq = sorted_ranges[0]
                start = seq[0]
                end = seq[-1]
                bill_file_list_front_trimmed = bill_file_list[:start]
                if "." in bill_file_list_front_trimmed:
                    index_period = bill_file_list_front_trimmed.index(".")
                    start = index_period + 1
                else:
                    start = 0
                end = min(end, start + 100)    
                bill_file_list_trimmed = bill_file_list[end:]
                if "." in bill_file_list_trimmed:
                    index_period = bill_file_list_trimmed.index(".")
                    end += index_period + 1
                else:
                    end = len(bill_file_list) - 1
                sentence = ' '.join(untouched_bill_text[start:end])
                summary_len = len(untouched_bill_text[start:end])
                
                assert(sentence == ' '.join(untouched_bill_text[start:end]))

                bill_text = ' '.join(untouched_bill_text)
                bill_len = len(untouched_bill_text)
                summary_text = sentence
                summary_len = len(untouched_bill_text[start:end])
                
                if len(bill_file_list[start:end]) < MIN_SUMMARY_LENGTH:
                    no_summaries.add(file_name)
                
                else:
                    #trim
                    if file_name not in no_summaries:
                        if bill_text not in already_seen:
                            already_seen.add(bill_text)
                            if start > 250:
                                #print bill_text
                                bill_new_start = start - 100
                                new_start_index = 100
                                bill_to_one_hundred = bill_text_list[bill_new_start:]
                                #now, find the first period
                                index_period = bill_to_one_hundred.index(".")
                                #greater to 100 means we can't trim
                                if index_period < 100:
                                    bill_new_start += index_period + 1
                                    new_start_index = new_start_index - index_period - 1

                                    bill_trimmed = bill_text_list[bill_new_start:]
                                    new_end_index = new_start_index + summary_len
                                    if(' '.join(bill_trimmed[new_start_index:new_end_index]) != ' '.join(untouched_bill_text[start:end])):
                                        print "WRONG"
                                        print bill_trimmed[new_start_index:new_end_index]
                                        print summary_text
                                        print
                                    bill_text = ' '.join(bill_trimmed)
                                    start = new_start_index
                                    end = new_end_index
                                    bill_len = len(bill_trimmed)

                                    assert(end - start + 1 == summary_len)
                                    assert(' '.join(bill_text.split()[start:end]) == summary_text)
                                    row = [file_name, bill_text, bill_len, summary_text, start, end, summary_len, '1']
                                    writer.writerow(row)
                                else:
                                    no_summaries.add(file_name)
                            elif bill_len > 3 * summary_len and bill_len > 500: #trim the end of the bill
                                new_end = int(random.uniform(0.3, 0.4) * bill_len)
                                new_end = max(end, new_end)
                                bill_file_list_trimmed = bill_file_list[new_end:]
                                if "." in bill_file_list_trimmed:
                                    index_period = bill_file_list_trimmed.index(".")
                                    new_end += index_period + 1
                                else:
                                    new_end = bill_len
                                bill_text_list = bill_text.split()
                                bill_text_list = bill_text_list[:new_end]
                                bill_len = len(bill_text_list)
                                bill_text = ' '.join(bill_text_list)
                                assert(' '.join(bill_text_list[start:end]) == summary_text)
                                row = [file_name, bill_text, bill_len, summary_text, start, end, summary_len, '1']
                                writer.writerow(row)
                            else:
                                row = [file_name, bill_text, bill_len, summary_text, start, end, summary_len]
                                writer.writerow(row)
                        else:
                            dupes += 1
                            print file_name
            else:
                no_summaries.add(file_name)

            bill.close()
            summary.close()          

            file_count += 1
            if file_count % 1000 == 0:
                print "finished ", file_count

        csvfile.close()

    print("Finished analyzing dataset!")
    print "no summaries for this many files", len(no_summaries)
    print "dupes: ", dupes

if __name__ == "__main__":
    main()