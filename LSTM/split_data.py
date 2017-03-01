import numpy as np
import random
import codecs
import os
import shutil
from os import listdir
from os import getcwd

# specify headline/article path here
headline_datapath = getcwd() + '/gold_summaries/'
article_datapath = getcwd() +'/cleaned_bills/'

def return_files(path):
    return [path+f for f in listdir(path) if (isfile(join(path, f)) and not f.startswith('.'))]

def return_dir(path):
    return [path+f for f in listdir(path) if (not f.startswith('.'))]

def split_dataset():
    #collect all of the paths
    dataset_len = 0
    file_names = []
    file_directories = return_dir(article_datapath)
    for dir_path in file_directories:
        dataset_len += len(os.listdir(dir_path))
        file_names += ([dir_path + '/' + name for name in os.listdir(dir_path)])

    print("There are %s bills" % dataset_len)
    # Random shuffle data
    random.seed(12)
    dataset = np.array(random.sample(file_names, dataset_len))

    #Split dataset to 70% training, 20% evaluation and 10% testing.
    train_size = int(dataset_len*0.7)
    eval_size = int(dataset_len*0.2)
    train, evalu, test = dataset[0:train_size], dataset[train_size:train_size+eval_size], dataset[train_size+eval_size:]
    return train, evalu, test

def gold_file_name(name):
    last_dash = name.rfind('_')
    file_body = name[:last_dash]
    gold_name = file_body + "_gold.txt"
    return gold_name

def write_encset(enc_train, enc_eval, enc_test): 
    # create and write training, evaluation, and testing encoding/decoding files.
    count = 0
    for name in enc_train:
        shutil.copy(name, './enc_train')
        #shutil.copy(headline_datapath + gold_file_name(name), './dec_train')
        count = count + 1

    for name in enc_eval:
        shutil.copy(name, './enc_eval')
        #shutil.copy(headline_datapath + gold_file_name(name), './dec_eval')
        count = count + 1

    for name in enc_test:
        shutil.copy(name, './enc_test')
        #shutil.copy(headline_datapath + gold_file_name(name), './dec_test')
        count = count + 1

    if count % 100 == 0:
        print "finished ", count

def main():
    enc_train, enc_eval, enc_test = split_dataset()
    write_encset(enc_train, enc_eval, enc_test)
    print("Finished splitting dataset!")

if __name__ == "__main__":
    main()
