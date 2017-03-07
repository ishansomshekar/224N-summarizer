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
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        index = haystack.find(needle, start+len(needle))
        if index > start:
            start = index
        n -= 1
    return start

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
    # with open('data_2x_greater_than_summaries.csv', 'wb') as csvfile:
    #     writer = csv.writer(csvfile)
    num_valid = 0
    for file_name, bill_adr_len in name_to_bill.items():
        summ_adr_len = name_to_summary[file_name]
        
        bill_adr = bill_adr_len[0]
        summ_adr = summ_adr_len[0]
        bill = open(bill_adr, 'r')
        bill_file_text = bill.read()
        # bill_stemmed = [stemmer.stem(word) for word in bill_file_text.split()]
        # bill_stemmed = ' '.join(bill_stemmed)
        # print bill_stemmed
        # print
        # print bill_file_text
        # print 
        # print ' '.join(bill_stemmed)
        # print

        # summ_file = open(summ_adr, 'r')
        # summ_text = summ_file.read()

        index = -1
        filter_length = 0
        bill_type_index = file_name.find("_")
        bill_type = file_name[:bill_type_index]
        if bill_type == 'hjres':
            index = bill_file_text.find("resolved by the senate and house of representatives of the united states of america in congress assembled")
            filter_length = len("resolved by the senate and house of representatives of the united states of america in congress assembled")
        elif bill_type == 'hconres':
            index = bill_file_text.find("resolved by the house of representatives ( the senate concurring )")
            filter_length = len("resolved by the house of representatives ( the senate concurring )")
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
            filter_length = len("be it resolved by the senate ( the house of representatives concurring )")
            index = bill_file_text.find("be it resolved by the senate ( the house of representatives concurring )")
        else: #sres
            filter_length = len("resolved")
            index = bill_file_text.find("resolved")
        
        if index > 0:
            #trim to first letter
            remaining_bill = bill_file_text[index + filter_length:]
            for idx, ch in enumerate(remaining_bill):
                if ch.isalpha():
                    remaining_bill = remaining_bill[idx:]
                    break
            index_period = find_nth(remaining_bill, ".", 5)
            remaining_bill = remaining_bill[:index_period + 1]
            print file_name
            print remaining_bill
            print
            num_valid += 1
        # summ_stemmed = [stemmer.stem(word) for word in summ_text.split()]
        # summ_stemmed = ' '.join(summ_stemmed)
        # print summ_text
        # print
        # print summ_stemmed
        # print
        # output_summary = ""

        # split_summ = summ_text.split()
        # # print split_summ
        # for i in xrange(0, len(split_summ)-8):
        #     val = bill_file_text.find(' '.join(split_summ[i:i+8]))
        #     # print ' '.join(split_summ[i:i+8])
        #     if val > 0:
        #         break
        #         print file_name
        #         print ' '.join(split_summ[i:i+8])
                #find_index = val

            # print find_index

        # for summ_line in summ_stemmed.split(". "):
        #     summ_line = summ_line.strip()
            # print summ_line
            # print
            # if bill_stemmed.find(summ_line) > 0:
            #     output_summary += summ_line + ". "

        
        # if output_summary != "":
        #     num_valid +=1
        #     print file_name
        #     print output_summary

        # index1 = bill_file_text.find("resolved by the senate")
        # index2 = bill_file_text.find("resolved by the house of representatives")
        # if index1 > 0 or index2 > 0:
            
        #     bill_summary = bill_file_text[index1:]
        #     # bill_sentences = bill_file_text.split(". ")
        #     # print file_name
        #     # print bill_summary
        #     # print
        #     num_valid += 1
        bill.close()

        # summ_file.seek(0)
        # summ_file.write(output_summary)
        # summ_file.truncate()
        #summ_file.close()
            

    #         find_index = bill_file_text.find(summ_file_text)



    #         #open the bill and summary files from bill_adr and summ_adr
    #         #FIND THE SUMMARY INSIDE THE BILL 
    #         #you can even do... trim the bill to 400 words, and then find the summary 
    #         #try printing out all the different start and end indices to make sure that this is still an interesting problem
    #         #i'd say... if we can get ~70,000

        file_count += 1
        if file_count % 1000 == 0:
            print "finished ", file_count

    #         threshold_bill_len = MAX_THRESHOLD * bill_len    
    #         if threshold_bill_len > summary_len and summary_len < 500 and bill_len < 1300 and bill_len > 0 and summary_len > 4 and find_index > 0:
    #             data_within_bounds += 1
    #             #we have approx 112,000 bills at this point. Now, run statistics to understand the median and such
    #             row = [file_name, bill_adr_len[0], bill_len, summ_adr_len[0], summary_len]        
    #             writer.writerow(row)

    # print data_within_bounds

    # print("Finished analyzing dataset!")
    print "num valid:, ", num_valid

if __name__ == "__main__":
    main()