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
    return ID


def process_file(path):
    fileList = return_files(path)
    newFilePath = path + '/processed/'

    body_titles = set(['   LEGISLATIVE COUNSEL\'S DIGEST\n'])

    total = len(fileList)
    count = 0
    for file in fileList:
        if count % 1000 == 0:
            print "finished %d" % count

        newFileName = get_bill_ID(file)
        newFile = open(newFilePath + newFileName + '.txt', 'w')
        with open(file,'r') as f:
            write = False
            switch = True
            for line in f:
                # print line
                # print
                print line
                if line == '    LEGISLATIVE COUNSEL\'S DIGEST\n':
                    print "found"
                    write = True               
                # if line in end_line:
                #     write = False
                # if line == '\n':
                #     switch = not switch
                if write and switch and line not in body_titles and line != '\n':
                    print line
                    line = line.strip()
                    if line.endswith('\n'): 
                        line = line[:-1]
                    line += ' '

                    newFile.write(line)
        count += 1     
        newFile.close()
        f.close()





def main(args):
    curDir = getcwd()
    txt_data_path = curDir + '/txt_files/'
    process_file(txt_data_path)


if __name__=='__main__':
    import sys
    main(sys.argv[1:])