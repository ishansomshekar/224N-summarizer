from os import listdir
from os.path import isfile, join, getsize
from collections import defaultdict
from collections import Counter
from os import getcwd
import os
import cProfile as cprofile
import xml.etree.cElementTree as ET
import string
import re
import csv

# This file reads in xml files in a directory 'xml_files' and 
# outputs txt files that have the bill ID as their name and 
# summary as their body to a directory 'processed'
#



def return_files(path):
    return [path+f for f in listdir(path) if (isfile(join(path, f)) and not f.startswith('.'))]

def get_bill_ID(file):
    names = file.split('/')
    ID = names[len(names)-1].split('.')[0]

    return ID


def process_xml(path, output_path):
    fileList = return_files(path)
    # print fileList
    newFilePath = path + 'processed/test/'
    printable = string.printable
    emptyFilePath = getcwd() + '/missing_files/'
    emptyFiles = []
    count = 0
    for file in fileList:
        newFileName = get_bill_ID(file)
        if count % 1000 == 0:
            print "finished %d" % count
        text = ''
        # print file
        tree = ET.parse(file)
        for elem in tree.iter(tag='summary'):
            text = elem.text
        if text == '':
            emptyFiles.append(get_bill_ID(file))
            continue

        text = text.replace(';', '.')
        text = text.strip()
        text = text.splitlines()
        text = text[1:]
        text = ''.join(text)
        text = re.sub(r'[^\x00-\x7F]+',' ', text)
        
        newFile = open(output_path + newFileName + '_gold.txt', 'w')
        newFile.write(text)
        newFile.close()

        count += 1
    emptyFile = open(emptyFilePath + "missing.txt", 'w')
    for file in emptyFiles:
        emptyFile.write(file + '\n')

def process_xml_subjects(file):
    keywords = []
    tree = ET.parse(file)
    for elem in tree.iter(tag='term'):
        keywords.append(elem.get('name'))
    return keywords






def main(args):
    # curDir = getcwd()
    # xml_data_path = curDir + '/xml_files/'
    # output_path = curDir + '/gold_summaries/'
    # # print xml_data_path
    # process_xml(xml_data_path, output_path)
    counter = 0
    with open('summary_words.csv', 'wb') as csvfile:
        writer = csv.writer(csvfile)

        for subdir, dirs, files in os.walk(getcwd()):
            for file in files:
                if counter % 1000 == 0:
                    print 'finished %d' % counter
                address = os.path.join(subdir, file)
                if (file.endswith('.xml')):
                    keywords = process_xml_subjects(address)

                    index_start = address.rfind('/') + 1
                    index_end = address.rfind('_')
                    file_name = address[index_start:index_end]
                    keywords.insert(0, file_name)
                    writer.writerow(keywords)
                counter += 1

    csvfile.close()
            

    # keywords = process_xml_subjects(getcwd() + '/sres_109_502.xml')
    # print keywords


if __name__=='__main__':
    import sys
    main(sys.argv[1:])