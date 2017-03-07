import numpy as np
from os import getcwd
from os import listdir
import csv
from nltk.stem.snowball import SnowballStemmer

bill_file = "bills_info.csv"
summary_file = "summaries_info.csv"

name_to_summary = dict()
name_to_bill = dict()

def find_nth(haystack, needle, n):
    haystack = ' '.join(haystack)
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        index = haystack.find(needle, start+len(needle))
        if index > start:
            start = index
        n -= 1
    previous_substr = haystack[:start + 1]
    #print previous_substr
    num_words = previous_substr.split()
    #num_words_preceding = len(haystack[:start].split())
    return len(num_words)

def main():
    stemmer = SnowballStemmer("english")
    print "generating dictionaries for bills and summaries"
    count = 0
    with open(bill_file, 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile.read().splitlines())]
        for row in rows:
            file_name = row[0]
            length = row[2]
            file = row[1]
            count += 1
            
            name_to_bill[file_name] = (file, length)

    with open(summary_file, 'rb') as csvfile:
        rows = [row for row in csv.reader(csvfile.read().splitlines())]
        for row in rows:
            file_name = row[0]
            length = row[2]
            file = row[1]
            count += 1
            
            name_to_summary[file_name] = (file, length)

    print "now reading dictionaries: "
    file_count = 0
    data_within_bounds = 0

    total_count = 0
    with open('bills_with_extracted.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for file_name, bill_adr_len in name_to_bill.items():
            
            bill_adr = bill_adr_len[0]
            bill = open(bill_adr, 'r')
            bill_file_text = bill.read()
            bill_file_text = bill_file_text.strip()
            bill_len = bill_adr_len[1]
            #print bill_len
            bill_file_text = bill_file_text.lower()
            total_count += 1
            index = -1
            filter_length = 0
            bill_type_index = file_name.find("_")
            bill_type = file_name[:bill_type_index]
            
            extracted_start = 0
            extracted_end = 0
            if bill_type == 'hjres':
                index = bill_file_text.find("resolved by the senate and house of representatives of the united states of america in congress assembled")
                filter_length = len("resolved by the senate and house of representatives of the united states of america in congress assembled")
            elif bill_type == 'hconres':
                index = bill_file_text.find("resolved by the house of representatives (the senate concurring)")
                filter_length = len("resolved by the house of representatives (the senate concurring)")
            elif bill_type == 'hr':
                index = bill_file_text.find("be it enacted by the senate and house of representatives of the united states of america in congress assembled")
                filter_length = len("be it enacted by the senate and house of representatives of the united states of america in congress assembled")
            elif bill_type == 'hres':
                index = bill_file_text.find("resolved")
                filter_length = len("resolved")
            elif bill_type == 's':
                filter_length = len("be it enacted by the senate and house of representatives of the united states of america in congress assembled")
                index = bill_file_text.find("be it enacted by the senate and house of representatives of the united states of america in congress assembled")
            elif bill_type == 'sconres':
                filter_length = len("be it resolved by the senate (the house of representatives concurring)")
                index = bill_file_text.find("be it resolved by the senate (the house of representatives concurring)")
            else: #sres
                filter_length = len("resolved")
                index = bill_file_text.find("resolved")
            
            if index > 0:
                #trim to first letter
                extracted_start_character = index + filter_length
                num_words_preceding = len(bill_file_text[:extracted_start_character].split())
                #print bill_file_text[:extracted_start_character].split()
                extracted_summary = bill_file_text[extracted_start_character:]
                extracted_summary = extracted_summary.split() 
                #print extracted_summary
                for idx, word in enumerate(extracted_summary):
                    if word.isalpha():
                        extracted_summary = extracted_summary[idx:]
                        num_words_preceding += idx
                        break
                #print extracted_summary
                # print extracted_summary 
                # print 
                # temp_bill = bill_file_text.split()
                # print ' '.join(temp_bill[extracted_start:])
                #trim to fifth period (ideally, the fifth sentence)
                extracted_end = num_words_preceding
                if bill_type == "s" or bill_type == "hr":
                    index_period = find_nth(extracted_summary, ".", 10)
                else:
                    index_period = find_nth(extracted_summary, ".", 5)
                #print index_period
                extracted_end += index_period

                
                extracted_summary_list = bill_file_text.split()[num_words_preceding : extracted_end]
                extracted_summary = ' '.join(extracted_summary_list)
                len_extracted = len(extracted_summary_list)
                bill_len = len(bill_file_text.split())

                if int(bill_len) < 1000 and len_extracted > 30:
                    row = [file_name, bill_file_text, bill_len, extracted_summary, num_words_preceding, extracted_end, len_extracted]
                    writer.writerow(row)
                    data_within_bounds += 1
          
            bill.close()

            file_count += 1
            if file_count % 1000 == 0:
                print "finished ", file_count

    print total_count
    print("Finished analyzing dataset!")
    print "data within bounds", data_within_bounds

if __name__ == "__main__":
    main()