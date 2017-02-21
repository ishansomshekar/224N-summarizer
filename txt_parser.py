from os import listdir
from os.path import isfile, join, getsize
from collections import defaultdict
from collections import Counter
from os import getcwd
import cProfile as cprofile
import xml.etree.cElementTree as ET


def return_files(path):
    return [path+f for f in listdir(path) if (isfile(join(path, f)) and not f.startswith('.'))]


def get_bill_ID(file):
    names = file.split('/')
    ID = names[len(names)-1].split('.')[0]
    print ID
    return ID



def process_file(path):
    fileList = return_files(path)
    print fileList
    newFilePath = path + '/processed/'

    body_titles = set(['A BILL',
     '                         CONCURRENT RESOLUTION\n',
     'RESOLUTION', 
     'JOINT RESOLUTION'])

    end_line = set(['                                 &lt;all&gt;\n'])

    for file in fileList:
        newFileName = get_bill_ID(file)
        newFile = open(newFilePath + newFileName + '.txt', 'w')
        with open(file,'r') as f:
            write = False
            switch = True
            for line in f:
                if line in body_titles:
                    write = True               
                if line in end_line:
                    write = False
                if line == '\n':
                    switch = not switch
                if write and switch and line not in body_titles and line != '\n':
                    newFile.write(line)
                
        newFile.close()
        f.close()





def main(args):
    curDir = getcwd()
    txt_data_path = curDir + '/txt_files/'
    print txt_data_path
    process_file(txt_data_path)


if __name__=='__main__':
    import sys
    main(sys.argv[1:])