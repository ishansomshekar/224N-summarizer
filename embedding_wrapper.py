from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import argparse

from six.moves import urllib

from tensorflow.python.platform import gfile
from tqdm import *
import numpy as np
from os.path import join as pjoin

import qa_data

def return_files(path):
    return [path+f for f in listdir(path) if (not f.startswith('missing_files') and not f.startswith('.'))]


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


    def build_vocab():
        dataset_len = 0
        file_names = []
        file_directories = return_dir(self.bills_datapath)
        for dir_path in file_directories:
            dataset_len += len(os.listdir(dir_path))
            file_names += return_files(dir_path)
        vocab = dict()
        reverse_vocab = []
        idx = 0
        wordcounter = 0
        for file in file_names:
            with open(file, 'r') as f:
                words = f.read().split()
                for word in words:
                    wordcounter += 1
                    if not word in vocab:
                        vocab[word] = idx
                        reverse_vocab +=[word]
                        idx += 1
        vocab['UNK'] = idx
        reverse_vocab += ['UNK']
        idx += 1
        vocab['PAD'] = idx
        reverse_vocab += ['PAD']
        wordcounter += 2

        self.vocab = vocab
        self.reverse_vocab = reverse_vocab
        self.num_tokens = wordcounter

        return self.vocab


    def process_glove():
        """
        :param vocab_list: [vocab]
        :return:
        """
        if not gfile.Exists(getcwd() + "trimmed_glove.6B.{}d.npz".format(self.glove_dim)):
            glove_path = os.path.join(getcwd(), "glove.6B.{}d.txt".format(self.glove_dim))
            glove = np.zeros((len(vocab_list), args.glove_dim))
            not_found = 0
            with open(glove_path, 'r') as fh:
                for line in tqdm(fh, total=size): #reading GLOVE line by line
                    array = line.lstrip().rstrip().split(" ")
                    word = array[0]
                    vector = list(map(float, array[1:]))
                    if word in self.vocab:
                        idx = self.vocab[word]
                        glove[idx, :] = vector
                    elif word.lower() in self.vocab:
                        idx = self.vocab[word.lower()]
                        glove[idx, :] = vector
                    else:
                        not_found += 1
            found = self.num_tokens - not_found
            print("{}/{} of word vocab have corresponding vectors in {}".format(found, self.num_tokens, glove_path))
            np.savez_compressed(save_path, glove=glove)
            print("saved trimmed glove matrix at: {}".format(save_path))

            self.embeddings = glove

if __name__ == '__main__':
    bills_datapath = getcwd() + '/ALL_CLEAN_BILLS/'
    gold_summaries_datapath = getcwd() +'/ALL_GOLD_SUMMARIES/'

    embedding_wrapper = EmbeddingWrapper(bills_datapath)
    embedding_wrapper.build_vocab()
    embedding_wrapper.process_glove()


