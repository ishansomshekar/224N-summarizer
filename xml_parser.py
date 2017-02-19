from os import listdir
from os.path import isfile, join, getsize
from collections import defaultdict
from collections import Counter
from os import getcwd
import cProfile as cprofile
import xml.etree.cElementTree as ET

# This file reads in xml files in a directory 'xml_files' and 
# outputs txt files that have the bill ID as their name and 
# summary as their body to a directory 'processed'
#



def return_files(path):
    return [path+f for f in listdir(path) if (isfile(join(path, f)) and not f.startswith('.'))]

def get_bill_ID(file):
	names = file.split('/')
	ID = names[len(names)-1].split('.')[0]
	print ID
	return ID


def process_xml(path):
	fileList = return_files(path)
	print fileList
	newFilePath = path + '/processed/'

	for file in fileList:
		text = ''
		print file
		tree = ET.parse(file)
		for elem in tree.iter(tag='summary'):
			text = elem.text
		newFileName = get_bill_ID(file)
		newFile = open(newFilePath + newFileName + '.txt', 'w')
		newFile.write(text)
		newFile.close()

def main(args):
    curDir = getcwd()
    xml_data_path = curDir + '/xml_files/'
    print xml_data_path
    process_xml(xml_data_path)


if __name__=='__main__':
    import sys
    main(sys.argv[1:])