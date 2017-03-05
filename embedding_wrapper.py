from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import argparse
import cPickle as pickle

from six.moves import urllib

from tensorflow.python.platform import gfile
from tqdm import *
import numpy as np
from os.path import join as pjoin

# import qa_data

def return_files(path):
    return [path+f for f in os.listdir(path) if (not f.startswith('missing_files') and not f.startswith('.'))]

def return_dir(path):
    return [path+f for f in os.listdir(path) if (not f.startswith('.'))]
    
class EmbeddingWrapper(object):
    def __init__(self, bills_datapath):     
        self.bills_datapath = bills_datapath
        self.vocab = None
        self.reverse_vocab = None
        self.embeddings = None
        self.glove_dir = "FILL GLOVE DIR HERE"
        self.glove_dim = 50
        self.num_tokens = 0
        self.file_names = []
        self.pad = 'PAD'
        self.unk = 'UNK'


    # def build_vocab(self):
    #     print ("building vocabulary for all files")
    #     dataset_len = 0
    #     file_names = []
    #     file_directories = return_dir(self.bills_datapath)
    #     for dir_path in file_directories:
    #         dataset_len += len(os.listdir(dir_path))
    #         file_names += return_files(dir_path + "/")
    #     vocab = dict()
    #     reverse_vocab = []
    #     idx = 0
    #     wordcounter = 0
    #     file_count = 0
    #     for file in file_names:
    #         with open(file, 'r') as f:
    #             words = f.read().split()
    #             for word in words:
    #                 wordcounter += 1
    #                 if not word in vocab:
    #                     vocab[word] = idx
    #                     reverse_vocab +=[word]
    #                     idx += 1
    #         file_count += 1
    #         if file_count % 1000 == 0:
    #         	print ("finished reading %d files" % file_count)
    #     vocab[self.unk] = idx
    #     reverse_vocab += [self.unk]
    #     idx += 1
    #     vocab[self.pad] = idx
    #     reverse_vocab += [self.pad]
    #     wordcounter += 2

    #     self.vocab = vocab
    #     self.reverse_vocab = reverse_vocab
    #     self.num_tokens = wordcounter
    #     print( "finished building vocabulary of size %d for all files" %wordcounter)

    def build_vocab(self):
        if not gfile.Exists(os.getcwd() + '/vocab.dat'):
            print ("building vocabulary for all files")
            dataset_len = 0
            vocab = dict()
            reverse_vocab = []
            idx = 0
            wordcounter = 0
            file_count = 0
            with open(self.bills_datapath, 'r') as f:
                for i, line in enumerate(f):
                    words = line.split()
                    for word in words:
                        wordcounter += 1
                        if not word in vocab:
                            vocab[word] = idx
                            reverse_vocab +=[word]
                            idx += 1
                    if i % 1000 == 0:
                        print ("finished reading %d bills" % i)

            vocab[self.unk] = idx
            reverse_vocab += [self.unk]
            idx += 1
            vocab[self.pad] = idx
            reverse_vocab += [self.pad]
            wordcounter += 2

            self.vocab = vocab
            self.reverse_vocab = reverse_vocab
            self.num_tokens = wordcounter
            print( "finished building vocabulary of size %d for all files" %wordcounter)
        else:
            self.vocab = pickle.load(open('vocab.dat', 'r'))
            self.reverse_vocab = pickle.load(open('reverse_vocab.dat', 'r'))
            self.num_tokens = len(self.vocab)



        #gemsim.models.word2vec 
    def process_glove(self, size = 4e5):
        """
        :param vocab_list: [vocab]
        :return:
        """
        save_path = os.getcwd() + "/trimmed_glove.6B.{}d.npz".format(self.glove_dim)
        if not gfile.Exists(save_path):
            print("build glove")
            glove_path = os.path.join(os.getcwd(), "glove.6B.{}d.txt".format(self.glove_dim))
            glove = np.zeros((len(self.vocab), self.glove_dim))
            not_found = 0
            found_words = []
            with open(glove_path, 'r') as fh:
                for line in tqdm(fh, total=size): #reading GLOVE line by line
                    array = line.lstrip().rstrip().split(" ")
                    word = array[0]
                    vector = list(map(float, array[1:]))
                    if word in self.vocab:
                        idx = self.vocab[word]
                        glove[idx, :] = vector
                        found_words.append(word)
                    elif word.lower() in self.vocab:
                        idx = self.vocab[word.lower()]
                        glove[idx, :] = vector
                        found_words.append(word)
                    else:
                        not_found += 1
            found = size - not_found
            #if the word isn't found, you word_vc = np.random.uniform(range, range, size)
            print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(self.vocab), glove_path))
            #print type(glove)
            #don't compress, might need a load compressed
            #bare bones, make sure it's the type that you want and make sure it save it without compressed
            np.savez_compressed(save_path, glove=glove)

            print("saved trimmed glove matrix at: {}".format(save_path))

            self.embeddings = glove

    def get_value(self, word):
        if word in self.vocab:
            return self.vocab[word]
        else:
            return self.vocab[self.unk]



if __name__ == '__main__':
    bills_datapath = os.getcwd() + '/bill_data_100.txt'
    gold_summaries_datapath = os.getcwd() +'/ALL_GOLD_SUMMARIES/'

    embedding_wrapper = EmbeddingWrapper(bills_datapath)
    embedding_wrapper.build_vocab()
    with open('vocab.dat', 'w') as f:
        pickle.dump(embedding_wrapper.vocab, f)
        f.close()

    dict_obj = pickle.load(open('vocab.dat', 'r'))

    with open('reverse_vocab.dat', 'w') as f:
        pickle.dump(embedding_wrapper.reverse_vocab, f)
        f.close()

    dict_obj = pickle.load(open('reverse_vocab.dat', 'r'))

    # assert dict_obj['games'] == embedding_wrapper.vocab['games']
    # print(dict_obj['games'])

    # print(embedding_wrapper.vocab)
    print
    # print(embedding_wrapper.reverse_vocab)
    print
    embedding_wrapper.process_glove()


