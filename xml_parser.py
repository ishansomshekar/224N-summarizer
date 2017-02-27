from os import listdir
from os.path import isfile, join, getsize
from collections import defaultdict
from collections import Counter
from os import getcwd
import cProfile as cprofile
import xml.etree.cElementTree as ET
import string
import re

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
    emptyFilePath = output_path + 'missing_files/'
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


def main(args):
    curDir = getcwd()
    xml_data_path = curDir + '/xml_files/'
    output_path = curDir + '/gold_summaries/'
    # print xml_data_path
    process_xml(xml_data_path, output_path)


if __name__=='__main__':
    import sys
    main(sys.argv[1:])