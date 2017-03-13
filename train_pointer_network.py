from embedding_wrapper import EmbeddingWrapper
from read_in_datafile import file_generator
import os
import numpy as np
from encoder_decoder_cells import DecoderCell
import heapq
import logging
import time
from attention_decoder import attention_decoder
from evaluate_prediction import normalize_answer
import rougescore as rs
from scipy.signal import argrelextrema
#from pointer_network import PointerCell

from util import Progbar

import tensorflow as tf

ATTENTION_FLAG = 1
UNIDIRECTIONAL_FLAG = True

logger = logging.getLogger("hw3.q2")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

t = time.localtime()
timeString  = time.strftime("%Y%m%d%H%M%S", t)
train_name = str(time.time())

class SequencePredictor():
    def __init__(self, embedding_wrapper):

        self.glove_dim = 50
        self.num_epochs = 10
        self.bill_length = 151
        self.keywords_length = 5
        self.lr = 0.0001
        self.inputs_placeholder = None
        self.summary_input = None
        self.mask_placeholder = None
        self.hidden_size = 20
        self.predictions = []
        self.batch_size = 5
        self.model_output = os.getcwd() + "model.weights"
        self.train_op = None
        self.loss = 0

        self.start_index_labels_placeholder = None
        self.end_index_labels_placeholder = None
        self.keywords_placeholder = None
        self.embedding_wrapper = embedding_wrapper
        self.vocab_size = embedding_wrapper.num_tokens
        self.embedding_init = None

        self.train_data_file = "bills_train_bills_3_150.txt" #"train_bills_3_context.txt"
        self.train_summary_data_file = "summaries_train_bills_3_150.txt"
        self.train_indices_data_file = "indices_train_bills_3_150.txt"
        self.train_sequence_data_file = "sequences_train_bills_3_150.txt"
        self.train_keyword_data_file = "train_bills_3_keywords.txt"
        file_open = open(self.train_data_file, 'r')
        self.train_len = len(file_open.read().split("\n"))
        file_open.close()

        self.dev_data_file =  "bills_dev_bills_3_150.txt"
        self.dev_summary_data_file =  "summaries_dev_bills_3_150.txt"
        self.dev_indices_data_file = "indices_dev_bills_3_150.txt"
        self.dev_sequence_data_file = "sequences_dev_bills_3_150.txt"
        self.dev_keyword_data_file = "dev_bills_3_keywords.txt"

        file_open = open(self.dev_data_file, 'r')
        self.dev_len = len(file_open.read().split("\n"))
        file_open.close()

    def batch_generator(self,embedding_wrapper, bill_data_path, indices_data_path, sequences_data_path, key_words_datapath, batch_size, MAX_BILL_LENGTH):

        f_generator = file_generator(batch_size, bill_data_path, indices_data_path, sequences_data_path, key_words_datapath)

        #pad the bills and summaries
        print "now padding and encoding batches"
        padded_bills = []
        padded_start_indices = []
        padded_end_indices = []
        padded_masks = []
        padded_keywords = []
        for bill_batch, indices_batch, sequences, keywords in f_generator:
            for idx, bill in enumerate(bill_batch):
                start_index, end_index = indices_batch[idx]
                sequence_len = sequences[idx]
                keywords_batch = keywords[idx]
                bill_list = [embedding_wrapper.get_value(word) for word in bill.split()]
                padded_keyword = [embedding_wrapper.get_value(word) for word in keywords_batch]
                # padded_summary = [embedding_wrapper.get_value(word) for word in summary] d g
                mask = [True] * min(len(bill_list), MAX_BILL_LENGTH)
                padded_bill = bill_list[:MAX_BILL_LENGTH]
                # padded_summary = padded_summary[:MAX_SUMMARY_LENGTH]
                mask = mask[:MAX_BILL_LENGTH]

                for i in xrange(0, MAX_BILL_LENGTH - len(padded_bill)):
                    padded_bill.append(embedding_wrapper.get_value(embedding_wrapper.pad))
                    mask.append(False)

                for i in xrange(0, 5 - len(padded_keyword)):
                    padded_keyword.append(embedding_wrapper.get_value(embedding_wrapper.pad))

                start_index_one_hot = [0] * MAX_BILL_LENGTH
                # if start_index >= MAX_BILL_LENGTH:
                #     start_index_one_hot[0] = 1
                #     start_index = 0
                # else:
                #     start_index_one_hot[start_index] = 1
                for i in xrange(start_index, end_index + 1):
                    start_index_one_hot[i] = 1

                #now pad start_index_one_hot starting at sequence_len to be alternating 0 and 1 to mask loss
                if (len(start_index_one_hot) > len(bill_list)):
                    val = 0
                    for i in xrange(0, len(start_index_one_hot) - sequence_len):
                        start_index_one_hot[sequence_len + i] = val
                        val ^= 1

                #generate normal distribution
                # distrib = np.random.normal(1.0, 0.5, int(.05 * len(start_index_one_hot)))
                # distrib = [x % 1. for x in distrib]
                distrib = [0.75, 0.5, 0.25]
                # print "distrb: "
                # print distrib
                distrib = sorted(distrib, reverse = True)
                #now, add around the one hot
                # for idx, value in enumerate(distrib):
                #     idx += 1
                #     if (start_index - idx) > 0 and (start_index - idx) < len(start_index_one_hot):
                #         start_index_one_hot[start_index - idx] = value
                #     if (start_index + idx) < len(start_index_one_hot):
                #         start_index_one_hot[start_index + idx] = value

                end_index_one_hot = [0] * MAX_BILL_LENGTH
                if end_index >= MAX_BILL_LENGTH:
                    end_index_one_hot[MAX_BILL_LENGTH - 1] = 1
                    end_index = MAX_BILL_LENGTH - 1
                else:
                    end_index_one_hot[end_index] = 1

                # for idx, value in enumerate(distrib):
                #     idx += 1
                #     if (end_index - idx) > 0 and (end_index - idx) < len(end_index_one_hot):
                #         end_index_one_hot[end_index - idx] = value
                #     if (end_index + idx) < len(end_index_one_hot):
                #         end_index_one_hot[end_index + idx] = value

                padded_masks.append(mask)
                padded_bills.append(padded_bill)
                padded_start_indices.append(start_index_one_hot)
                padded_end_indices.append(end_index_one_hot)
                padded_keywords.append(padded_keyword)

            yield padded_bills, padded_start_indices, padded_end_indices, padded_masks, sequences, padded_keywords
            print padded_start_indices
            # print padded_end_indices
            padded_bills = []
            padded_start_indices = []
            padded_end_indices = []
            padded_masks = []
            padded_keywords = []

    def create_feed_dict(self, inputs_batch, masks_batch, sequences, keywords_batch, start_labels_batch = None, end_labels_batch = None):
        feed_dict = {
            self.inputs_placeholder : inputs_batch,
            self.mask_placeholder : masks_batch,
            self.sequences_placeholder : sequences,
            self.keywords_placeholder : keywords_batch
            }
        if start_labels_batch is not None:
            feed_dict[self.start_index_labels_placeholder] = start_labels_batch
        if end_labels_batch is not None:
            feed_dict[self.end_index_labels_placeholder] = end_labels_batch
        return feed_dict

    def add_placeholders(self):
        self.inputs_placeholder = tf.placeholder(tf.int32, shape=(None, self.bill_length))
        self.mask_placeholder = tf.placeholder(tf.bool, shape=(None, self.bill_length))
        self.start_index_labels_placeholder = tf.placeholder(tf.float64, shape=(None, self.bill_length))
        self.end_index_labels_placeholder = tf.placeholder(tf.float64, shape=(None,self.bill_length))
        self.sequences_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size))
        self.keywords_placeholder = tf.placeholder(tf.int32, shape=(None, self.keywords_length))

    def return_embeddings(self):
        data = np.load('trimmed_glove.6B.50d.npz')
        embeddings = tf.Variable(data['glove'])
        bill_embeddings = tf.nn.embedding_lookup(embeddings, self.inputs_placeholder)
        bill_embeddings = tf.reshape(bill_embeddings, (-1, self.bill_length, self.glove_dim))
        keywords_embeddings = tf.nn.embedding_lookup(embeddings, self.keywords_placeholder)
        keywords_embeddings = tf.reshape(keywords_embeddings, (-1, self.keywords_length, self.glove_dim))
        return bill_embeddings, keywords_embeddings

    # def add_unidirectional_prediction_op(self, bill_embeddings):          
    #     #use hidden states in encoder to make a predictions
    #     with tf.variable_scope("encoder"):
    #         enc_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
    #         outputs, state = tf.nn.dynamic_rnn(enc_cell,bill_embeddings, dtype = tf.float64) #outputs is (batch_size, bill_length, hidden_size)
        
    #     with tf.variable_scope("backwards_encoder"):
    #         dims = [False, False, True]
    #         reverse_embeddings = tf.reverse(bill_embeddings, dims) 
    #         bck_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
    #         b_outputs, b_state = tf.nn.dynamic_rnn(bck_cell,reverse_embeddings, dtype = tf.float64) #outputs is (batch_size, bill_length, hidden_size)
        
    #     complete_outputs = tf.concat(2, [outputs, b_outputs]) #h_t is (batch_size, hidden_size *2 )

    #     preds_start = []
    #     preds_end = []
    #     with tf.variable_scope("decoder"):
    #         U_1_start = tf.get_variable('U_1_start', (self.hidden_size * 2,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
    #         U_1_end = tf.get_variable('U_1_end', (self.hidden_size * 2,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
    #         b2_1 = tf.get_variable('b2_1', (self.bill_length,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
    #         b2_2 = tf.get_variable('b2_2', (self.bill_length,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
    #         for i in xrange(self.batch_size):
    #             bill = complete_outputs[i, :, :] #bill is bill_length by hidden_size
    #             result_start = tf.matmul(bill, U_1_start) + b2_1
    #             result_end = tf.matmul(bill, U_1_end) + b2_2
    #             preds_start.append(result_start)
    #             preds_end.append(result_end)
    #     preds_start = tf.pack(preds_start)
    #     preds_end = tf.pack(preds_end)
    #     preds_start = tf.squeeze(preds_start)
    #     preds_end = tf.squeeze(preds_end)

    #     self.predictions = preds_start, preds_end
    #     return preds_start, preds_end

    def add_pointer_prediction_op(self, bill_embeddings):          
        #use hidden states in encoder to make a predictions
        forward_hidden_states = []
        # initial_state = tf.nn.rnn_cell.RNNCell.zero_state(self.batch_size, dtype=tf.float64)

        with tf.variable_scope("encoder"):
            enc_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            initial_state = enc_cell.zero_state(self.batch_size, dtype=tf.float64)
            for time_step in xrange(self.bill_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                o_t, h_t = enc_cell(bill_embeddings[:, time_step, :], initial_state)
                forward_hidden_states.append(h_t)
            #outputs, state = tf.nn.dynamic_rnn(enc_cell,bill_embeddings, dtype = tf.float64) #outputs is (batch_size, bill_length, hidden_size)
        backwards_hidden_states = []
        # initial_state = tf.nn.rnn_cell.RNNCell.zero_state(self.batch_size, dtype=tf.float64)
        with tf.variable_scope("backwards_encoder"):
            dims = [False, False, True]
            reverse_embeddings = tf.reverse(bill_embeddings, dims) 
            bck_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            initial_state = bck_cell.zero_state(self.batch_size, dtype=tf.float64)
            for time_step in xrange(self.bill_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                o_t, h_t = bck_cell(reverse_embeddings[:, time_step, :], initial_state)
                backwards_hidden_states.append(h_t)
        
        forward_hidden_states = [tf.concat(1, [hidden_state[0], hidden_state[1]]) for hidden_state in forward_hidden_states]
        forward_hidden_states = tf.pack(forward_hidden_states) #should now be (batch_size, bill_length, hidden_size)
        forward_hidden_states = tf.transpose(forward_hidden_states, [1, 0, 2])
        backwards_hidden_states = [tf.concat(1, [hidden_state[0], hidden_state[1]]) for hidden_state in backwards_hidden_states]
        backwards_hidden_states = tf.pack(backwards_hidden_states) #should now be (batch_size, bill_length, hidden_size)
        backwards_hidden_states = tf.transpose(backwards_hidden_states, [1, 0, 2])
        complete_hidden_states = tf.concat(2, [forward_hidden_states, backwards_hidden_states] ) #should be (batch_size, bill_length, hiddensize * 4 )
        # print forward_hidden_states
        # print backwards_hidden_states
        #print complete_hidden_states

        preds_start = []
        preds_end = []
        with tf.variable_scope("decoder"):
            # tf.get_variable_scope().reuse_variables() doesnt work because of first pass through loop
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size*4)
            state = cell.zero_state(self.batch_size, dtype=tf.float64)
            W1_start = tf.get_variable('W1_start', (self.batch_size, self.batch_size), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64) 
            W2_start = tf.get_variable('W2_start', (self.batch_size, self.batch_size), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
            vt_start = tf.get_variable('vt_start', (self.hidden_size * 4,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)

            W1_end = tf.get_variable('W1_end', (self.batch_size, self.batch_size), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64) 
            W2_end = tf.get_variable('W2_end', (self.batch_size, self.batch_size), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
            vt_end = tf.get_variable('vt_end', (self.hidden_size * 4,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
            # b2_1 = tf.get_variable('b2_1', (self.bill_length,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
            # b2_2 = tf.get_variable('b2_2', (self.bill_length,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)            
            for time_step in xrange(self.bill_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
               
                o_t, h_t = cell(bill_embeddings[:, time_step, :], state) # o_t is batch_size, 1
                
                x_start = tf.matmul(W1_start, complete_hidden_states[:, time_step, :]) # result is 1 , hidden_size*4
                y_start = tf.matmul(W2_start, o_t) # result is 1 , hidden_size*4
                u_start = tf.nn.tanh(x_start + y_start) #(batch_size, hidden_size * 4)
                p_start = tf.matmul(u_start, vt_start) #(batch_size, bill_length)

                x_end = tf.matmul(W1_end, complete_hidden_states[:, time_step, :]) # result is 1 , hidden_size*4
                y_end = tf.matmul(W2_end, o_t) # result is 1 , hidden_size*4
                u_end = tf.nn.tanh(x_end + y_end) #(batch_size, hidden_size * 4)
                p_end = tf.matmul(u_end, vt_end) #(batch_size, bill_length)

                preds_start.append(tf.nn.softmax(p_start))
                preds_end.append(tf.nn.softmax(p_end))
            tf.get_variable_scope().reuse_variables() # set here for each of the next epochs //not working
            assert tf.get_variable_scope().reuse == True

        preds_start = tf.pack(preds_start)
        preds_start = tf.squeeze(preds_start)
        preds_start = tf.transpose(preds_start,[1,0])
        preds_end = tf.pack(preds_end)
        preds_end = tf.squeeze(preds_end)
        preds_end = tf.transpose(preds_end,[1,0])
        self.predictions = (preds_start, preds_end)
        return (preds_start, preds_end)

    def add_loss_op(self, preds):
        #loss_1 = tf.nn.softmax_cross_entropy_with_logits(preds[0], self.start_index_labels_placeholder)
        #loss_2 = tf.nn.softmax_cross_entropy_with_logits(preds[1], self.end_index_labels_placeholder)
        loss_1 = tf.nn.l2_loss(preds[0] - self.start_index_labels_placeholder)
        #loss_2 = tf.nn.l2_loss(preds[1] - self.end_index_labels_placeholder)
        # masked_loss = tf.boolean_mask(loss_1, self.mask_placeholder)
        loss = loss_1 #+ loss_2
        self.loss = tf.reduce_mean(loss)   
        return self.loss

    def add_optimization(self, losses):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(losses)
        return self.train_op    

    def output(self, sess):
        start_preds = []
        end_preds = []
        prog = Progbar(target=1 + int(self.dev_len/ self.batch_size))
        count = 0
        for inputs,start_index_labels,end_index_labels, masks, sequences, keywords in self.batch_generator(self.embedding_wrapper, self.dev_data_file, self.dev_indices_data_file, self.dev_sequence_data_file, self.dev_keyword_data_file, self.batch_size, self.bill_length):
            start_, end_ = self.predict_on_batch(sess, inputs, start_index_labels, end_index_labels, masks, sequences, keywords)
            start_preds += start_.tolist()
            end_preds += end_.tolist()
            prog.update(count + 1, [])
            count +=1
        return zip(start_preds, end_preds)

    def evaluate_two_hots(self, sess):
        correct_preds, total_correct, total_preds, number_indices = 0., 0., 0., 0.
        start_num_exact_correct, end_num_exact_correct = 0, 0
        gold_standard_summaries = open(self.dev_data_file, 'r')
        gold_indices = open(self.dev_indices_data_file, 'r')
        file_name = train_name + "/" + str(time.time()) + ".txt"
        with open(file_name, 'a') as f:
            for start_preds, end_preds in self.output(sess):
                print "start preds: "
                print start_preds
                print "end preds: "
                print end_preds
                gold = gold_indices.readline()
                # print "gold before" 
                # print gold
                gold = gold.split()
                # print "gold split"
                # print gold
                gold_start = int(gold[0])
                gold_end = int(gold[1])

                np_start_preds = np.asarray(start_preds)
                start_maxima = argrelextrema(np_start_preds, np.greater)[0]
                # print "###########"
                # print start_maxima
                tuples = [(x, np_start_preds[x]) for x in start_maxima]
                # print tuples
                start_maxima = sorted(tuples, key = lambda x: x[1])
                # print maxima
                if len(start_maxima) > 0:
                    start_index = start_maxima[-1][0]
                else:
                    start_index = start_preds.index(max(start_preds))

                np_end_preds = np.asarray(end_preds)
                end_maxima = argrelextrema(np_end_preds, np.greater)[0]
                # print "###########"
                # print end_maxima
                tuples = [(x, np_end_preds[x]) for x in end_maxima]
                # print tuples
                end_maxima = sorted(tuples, key = lambda x: x[1])
                # print maxima
                if len(end_maxima) > 0:
                    end_index = end_maxima[-1][0]
                else:
                    end_index = end_preds.index(max(end_preds))

                print
                print "gold start ", (gold_start)
                print "our start " , (start_index)
                print "gold end ", (gold_end)
                print "our end ", (end_index)

                text = gold_standard_summaries.readline()
                summary = ' '.join(text.split()[start_index:end_index])
                gold_summary = ' '.join(text.split()[gold_start:gold_end])
                summary = normalize_answer(summary)
                gold_summary = normalize_answer(gold_summary)

                f.write(summary + ' \n')
                f.write(gold_summary + ' \n')

                x = range(start_index,end_index + 1)
                y = range(gold_start,gold_end + 1)
                xs = set(x)
                overlap = xs.intersection(y)
                overlap = len(overlap)

                if start_index == gold_start:
                    start_num_exact_correct += 1
                if end_index == gold_end:
                    end_num_exact_correct += 1
                
                number_indices += 1
                correct_preds += overlap
                total_preds += len(x)
                total_correct += len(y)

            start_exact_match = start_num_exact_correct/number_indices
            end_exact_match = end_num_exact_correct/number_indices
            p = correct_preds / total_preds if correct_preds > 0 else 0
            r = correct_preds / total_correct if correct_preds > 0 else 0
            f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

            gold_indices.close()

            f.write('Model results: \n')
            f.write('learning rate: %d \n' % self.lr)
            f.write('batch size: %d \n' % self.batch_size)
            f.write('hidden size: %d \n' % self.hidden_size)
            f.write('bill_length: %d \n' % self.bill_length)
            f.write('bill_file: %s \n' % self.train_data_file)
            f.write('dev_file: %s \n' % self.dev_data_file)
            f.write("Epoch start_exact_match/end_exact_match/P/R/F1: %.2f/%.2f/%.2f/%.2f/%.2f \n" % (start_exact_match, end_exact_match, p, r, f1))
            f.close()
        
        return (start_exact_match, end_exact_match), (p, r, f1)



    def evaluate_one_hot(self, sess):
        correct_preds, total_correct, total_preds, number_indices = 0., 0., 0., 0.
        start_num_exact_correct, end_num_exact_correct = 0, 0
        gold_standard = open(self.dev_indices_data_file, 'r')
        file_dev = open(self.dev_data_file, 'r')
        file_name = train_name + "/" + str(time.time()) + ".txt"
        # rouge_scores = []
        with open(file_name, 'w') as f:
            for batch_preds in self.output(sess):
                for preds in batch_preds:
                    index_prediction = preds
                    gold = gold_standard.readline()
                    gold = normalize_answer(gold)
                    gold = gold.split()
                    gold_start = int(gold[0])
                    gold_end = int(gold[1])

                    index_prediction = index_prediction.tolist()
                    maxStart = max(index_prediction)
                    index_max1 = index_prediction.index(maxStart)
                    
                    index_prediction_copy = index_prediction[:]
                    index_prediction_copy[index_max1] = 0
                    maxEnd = max(index_prediction_copy)
                    index_max2 = index_max1 + 20 #index_prediction_copy.index(maxEnd)

                    #switch the orders if necessary
                    start_index = min(index_max2, index_max1)
                    end_index = max(index_max2, index_max1)

                    text = file_dev.readline()
                    summary = ' '.join(text.split()[start_index:end_index])
                    gold_summary = ' '.join(text.split()[gold_start:gold_end])
                    f.write(summary + ' \n')
                    f.write(gold_summary + ' \n')
                    # if gold_summary != '':
                    #     rouge_l_score = rs.rouge_l(summary, [gold_summary], 0.5)
                    #     rouge_scores.append(rouge_l_score)                      

                    # print "our start guess: ", start_index
                    # print "gold_start: ", gold_start
                    # print "our end guess: ", end_index
                    # print "gold_end: ", gold_end

                    x = range(start_index,end_index + 1)
                    y = range(gold_start,gold_end + 1)
                    xs = set(x)
                    overlap = xs.intersection(y)
                    overlap = len(overlap)

                    if start_index == gold_start:
                        start_num_exact_correct += 1
                    if end_index == gold_end:
                        end_num_exact_correct += 1
                    
                    number_indices += 1
                    correct_preds += overlap
                    total_preds += len(x)
                    total_correct += len(y)

            start_exact_match = start_num_exact_correct/number_indices
            end_exact_match = end_num_exact_correct/number_indices
            p = correct_preds / total_preds if correct_preds > 0 else 0
            r = correct_preds / total_correct if correct_preds > 0 else 0
            f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

            gold_standard.close()
            # avg_rouge = np.mean(rouge_scores)

            f.write('Model results: \n')
            f.write('learning rate: %.2f \n' % self.lr)
            f.write('batch size: %d \n' % self.batch_size)
            f.write('hidden size: %d \n' % self.hidden_size)
            f.write('bill_length: %d \n' % self.bill_length)
            f.write('bill_file: %s \n' % self.train_data_file)
            f.write('dev_file: %s \n' % self.dev_data_file)
            f.write("Epoch start_exact_match/end_exact_match/P/R/F1: %.2f/%.2f/%.2f/%.2f/%.2f \n" % (start_exact_match, end_exact_match, p, r, f1))
            # f.write('avg rouge: %.3f \n' % avg_rouge)
            f.close()
        
        return (start_exact_match, end_exact_match), (p, r, f1)
    
    def predict_on_batch(self, sess, inputs_batch, start_index_labels, end_index_labels, mask_batch, sequence_batch, keywords_batch):
        feed = self.create_feed_dict(inputs_batch = inputs_batch, start_labels_batch=start_index_labels, masks_batch=mask_batch, sequences = sequence_batch, keywords_batch = keywords_batch, end_labels_batch = end_index_labels)
        predictions = sess.run(self.predictions, feed_dict=feed)
        # print predictions
        return predictions

    def train_on_batch(self, sess, inputs_batch, start_labels_batch, end_labels_batch, mask_batch, sequence_batch, keywords_batch):
        #print start_labels_batch
        feed = self.create_feed_dict(inputs_batch = inputs_batch, start_labels_batch=start_labels_batch, masks_batch=mask_batch, sequences = sequence_batch, keywords_batch = keywords_batch, end_labels_batch = end_labels_batch)
        ##### THIS IS SO CONFUSING ######
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, sess):
        prog = Progbar(target=1 + int(self.train_len / self.batch_size))
        count = 0
        for inputs,start_labels, end_labels, masks, sequences, keywords in self.batch_generator(self.embedding_wrapper, self.train_data_file, self.train_indices_data_file, self.train_sequence_data_file, self.train_keyword_data_file, self.batch_size, self.bill_length):
            tf.get_variable_scope().reuse_variables()
            loss = self.train_on_batch(sess, inputs, start_labels, end_labels, masks, sequences, keywords)
            prog.update(count + 1, [("train loss", loss)])
            count += 1
        print("")

        print("Evaluating on development data")
        exact_match, entity_scores = self.evaluate_two_hots(sess)
        print("Entity level end_exact_match/start_exact_match/P/R/F1: %.2f/%.2f/%.2f/%.2f", exact_match[0], exact_match[1], entity_scores[0], entity_scores[1], entity_scores[2])

        f1 = entity_scores[-1]
        return f1
    
    def fit(self, sess, saver):
        best_score = 0.
        epoch_scores = []
        for epoch in range(self.num_epochs):
            tf.get_variable_scope().reuse_variables()
            print("Epoch %d out of %d" % (epoch + 1, self.num_epochs))
            score = self.run_epoch(sess)
            if score > best_score:
                best_score = score
                if saver:
                    print("New best score! Saving model in %s" % self.model_output)
                    saver.save(sess, self.model_output)
            epoch_scores.append(score)
            print("")

    def initialize_model(self):
        self.add_placeholders()
        bill_embeddings, keywords_embeddings = self.return_embeddings()
        logger.info("Running attentive...",)
        preds = self.add_pointer_prediction_op(bill_embeddings)
        loss = self.add_loss_op(preds)
        self.train_op = self.add_optimization(loss)
        return preds, loss, self.train_op

def build_model(embedding_wrapper):
    with tf.variable_scope('attentive_model'):
        logger.info("Building model...",)
        start = time.time()
        model = SequencePredictor(embedding_wrapper)
        preds, loss, train_op = model.initialize_model()
        logger.info("took %.2f seconds", time.time() - start)

        tf.get_variable_scope().reuse_variables()
        init = tf.global_variables_initializer()
        tf.get_variable_scope().reuse_variables()
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver)

def main():
    mydir = os.path.join(os.getcwd(), train_name)
    os.makedirs(mydir)

    embedding_wrapper = EmbeddingWrapper()
    embedding_wrapper.build_vocab()
    embedding_wrapper.process_glove()
    build_model(embedding_wrapper)



if __name__ == "__main__":
    main()


