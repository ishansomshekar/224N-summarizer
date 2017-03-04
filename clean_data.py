import numpy as np
from os import getcwd
from os import listdir
import re
import csv

datapaths = [getcwd() +'/ALL_CLEAN_BILLS/', getcwd() +'/ALL_GOLD_SUMMARIES/']

def return_files(path):
    return [path+f for f in listdir(path) if (not f.startswith('missing_files') and not f.startswith('.'))]

def return_dir(path):
    return [path+f for f in listdir(path) if (not f.startswith('.'))]

def main():
    dataset_len = 0
    file_count = 0
    for path in datapaths:
        print "reading in: " + path
        file_directories = return_dir(path)

        for dir_path in file_directories:
            print "reading directory: ", dir_path
            dataset_len += len(listdir(dir_path))
            file_names = return_files(dir_path + '/')
            for file in file_names:
                file_obj = open(file, "r+")
                file_text = file_obj.read()
                file_text = file_text.replace("\n", "")
                file_text = re.sub("S(e|E)(c|C). [0-9]+.", "", file_text)

                file_text = re.sub("\([0-9]\)", "", file_text)
                file_text = re.sub("\((iv|v?i{0,3})\)", "", file_text)
                file_text = re.sub("--", "", file_text)
                file_text = re.sub("\([A-Za-z]\)", "", file_text)
                file_text = re.sub("[0-9]", "#", file_text)
                file_text = re.sub("#(#|-)+", "#", file_text)
                file_text = re.sub("([A-Z]{2,})", "", file_text)
                file_text = re.sub("(\.)+", '.', file_text)
                file_text = re.sub('([;\.,\[\]\-!\?\(\)])', r' \1 ', file_text)

                file_text = file_text.lower()
                file_text = re.sub("table of contents.", "", file_text)
                file_text = re.sub("short title.", "", file_text)

                file_text = re.sub("( )+", ' ', file_text)
                file_text = re.sub("\. \.", ".", file_text)
                file_text = re.sub("\'\'", "", file_text)
                file_text = re.sub("`", "", file_text)
                file_text = re.sub("\'s", " \'s", file_text)
                file_text = re.sub("whereas ", "", file_text)

                file_obj.seek(0)
                file_obj.write(file_text)
                file_obj.truncate()
                file_obj.close()

                file_count += 1
                
                if file_count % 1000 == 0:
                    print "finished ", file_count

if __name__ == "__main__":
    main()