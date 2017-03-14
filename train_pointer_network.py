from embedding_wrapper import EmbeddingWrapper
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
train_name = "pointername" + str(time.time())
logs_path = os.getcwd() + '/tf_log/'
train = True

class SequencePredictor():
    def __init__(self, embedding_wrapper):

        self.glove_dim = 50
        self.num_epochs = 10
        self.bill_length = 151
        self.keywords_length = 5
        self.lr = 0.0001
        self.inputs_placeholder = None
        self.summary_input = None
        self.summary_op = None
        self.mask_placeholder = None
        self.hidden_size = 20
        self.predictions = []
        self.batch_size = 10
        self.model_output = os.getcwd() + "model.weights"
        self.train_op = None
        self.loss = 0
        self.writer = None
        self.dropout = 0.5

        self.start_index_labels_placeholder = None
        self.end_index_labels_placeholder = None
        self.keywords_placeholder = None
        self.embedding_wrapper = embedding_wrapper
        self.vocab_size = embedding_wrapper.num_tokens
        self.embedding_init = None

        self.train_data_file = "bills_train_bills_4_150.txt" #"train_bills_3_context.txt"
        self.train_summary_data_file = "summaries_train_bills_4_150.txt"
        self.train_indices_data_file = "indices_train_bills_4_150.txt"
        self.train_sequence_data_file = "sequences_train_bills_4_150.txt"
        self.train_keyword_data_file = "train_bills_3_keywords.txt"
        file_open = open(self.train_data_file, 'r')
        self.train_len = len(file_open.read().split("\n"))
        file_open.close()

        self.dev_data_file =  "bills_dev_bills_4_150.txt"
        self.dev_summary_data_file =  "summaries_dev_bills_4_150.txt"
        self.dev_indices_data_file = "indices_dev_bills_4_150.txt"
        self.dev_sequence_data_file = "sequences_dev_bills_4_150.txt"
        self.dev_keyword_data_file = "dev_bills_3_keywords.txt"

        self.test_data_file =  "bills_test_bills_4_150.txt"
        self.test_summary_data_file =  "summaries_test_bills_4_150.txt"
        self.test_indices_data_file = "indices_test_bills_4_150.txt"
        self.test_sequence_data_file = "sequences_test_bills_4_150.txt"
        self.test_keyword_data_file = "test_bills_3_keywords.txt"


        file_open = open(self.dev_data_file, 'r')
        self.dev_len = len(file_open.read().split("\n"))
        file_open.close()


    def batch_gen_test(self, embedding_lookup, bill_data_path):
        
        current_batch_bills = []
        with tf.gfile.GFile(bill_data_path, mode="r") as source_file:
            for bill in source_file:
                bill_list = [self.embedding_wrapper.get_value(word) for word in bill.split()]
                padded_bill = bill_list[:self.bill_length]
                for i in xrange(0, self.bill_length - len(padded_bill)):
                    padded_bill.append(self.embedding_wrapper.get_value(self.embedding_wrapper.pad))

                current_batch_bills.append(padded_bill)
                if len(current_batch_bills) == self.batch_size:
                    yield current_batch_bills
                    current_batch_bills = []


    def file_generator(self, batch_size, bill_data_path, indices_data_path, sequences_data_path, keywords_data_path):    
        current_batch_summaries = []
        current_batch_bills = []
        current_batch_sequences = []
        current_batch_keywords = []
        counter = 0
        with tf.gfile.GFile(bill_data_path, mode="r") as source_file:
            with tf.gfile.GFile(indices_data_path, mode="r") as target_file:
                with tf.gfile.GFile(sequences_data_path, mode="r") as seq_file:
                    for bill in source_file:
                        indices = target_file.readline()
                        sequence_len = seq_file.readline()
                        counter += 1
                        start_and_end = indices.split()
                        current_batch_bills.append(bill)
                        # keywords = keyword_file.readline()
                        # keywords_list = keywords.split()
                        current_batch_summaries.append((int(start_and_end[0]), int(start_and_end[1])))
                        current_batch_sequences.append(int(sequence_len))
                        # current_batch_keywords.append(keywords_list)
                        if len(current_batch_summaries) == batch_size:
                            yield current_batch_bills, current_batch_summaries, current_batch_sequences, current_batch_keywords
                            current_batch_bills = []
                            current_batch_summaries = []
                            current_batch_sequences = []
                            current_batch_keywords = []

    def batch_generator(self,embedding_wrapper, bill_data_path, indices_data_path, sequences_data_path, key_words_datapath, batch_size, MAX_BILL_LENGTH):

        f_generator = self.file_generator(batch_size, bill_data_path, indices_data_path, sequences_data_path, key_words_datapath)

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
                #keywords_batch = keywords[idx]
                bill_list = [embedding_wrapper.get_value(word) for word in bill.split()]
                #padded_keyword = [embedding_wrapper.get_value(word) for word in keywords_batch]
                # padded_summary = [embedding_wrapper.get_value(word) for word in summary] d g
                mask = [True] * min(len(bill_list), MAX_BILL_LENGTH)
                padded_bill = bill_list[:MAX_BILL_LENGTH]
                # padded_summary = padded_summary[:MAX_SUMMARY_LENGTH]
                mask = mask[:MAX_BILL_LENGTH]

                for i in xrange(0, MAX_BILL_LENGTH - len(padded_bill)):
                    padded_bill.append(embedding_wrapper.get_value(embedding_wrapper.pad))
                    mask.append(False)

                # for i in xrange(0, 5 - len(padded_keyword)):
                #     padded_keyword.append(embedding_wrapper.get_value(embedding_wrapper.pad))

                start_index_one_hot = [0] * MAX_BILL_LENGTH
                if start_index >= MAX_BILL_LENGTH:
                    start_index_one_hot[0] = 1
                    start_index = 0
                else:
                    start_index_one_hot[start_index] = 1

                end_index_one_hot = [0] * MAX_BILL_LENGTH
                if end_index >= MAX_BILL_LENGTH:
                    end_index_one_hot[MAX_BILL_LENGTH - 1] = 1
                    end_index = MAX_BILL_LENGTH - 1
                else:
                    end_index_one_hot[end_index] = 1

                padded_masks.append(mask)
                padded_bills.append(padded_bill)
                padded_start_indices.append(start_index_one_hot)
                padded_end_indices.append(end_index_one_hot)
                #padded_keywords.append(padded_keyword)

            yield padded_bills, padded_start_indices, padded_end_indices, padded_masks, sequences, padded_keywords
            padded_bills = []
            padded_start_indices = []
            padded_end_indices = []
            padded_masks = []
            #padded_keywords = []

    def create_feed_dict(self, inputs_batch, masks_batch=None, sequences=None, keywords_batch=None, start_labels_batch = None, end_labels_batch = None):
        feed_dict = {
            self.inputs_placeholder : inputs_batch
            }
        if masks_batch is not None:
            feed_dict[self.mask_placeholder] = masks_batch
        if sequences is not None:
            feed_dict[self.sequences_placeholder] = sequences
        if keywords_batch is not None:
            feed_dict[self.keywords_placeholder] = keywords_batch
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
            W1_start = tf.get_variable('W1_start', (self.batch_size, self.batch_size), initializer = tf.constant_initializer(np.eye(self.batch_size)), dtype = tf.float64) 
            W2_start = tf.get_variable('W2_start', (self.batch_size, self.batch_size), initializer = tf.constant_initializer(np.eye(self.batch_size)), dtype = tf.float64)
            vt_start = tf.get_variable('vt_start', (self.hidden_size * 4,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)

            W1_end = tf.get_variable('W1_end', (self.batch_size, self.batch_size), initializer = tf.constant_initializer(np.eye(self.batch_size)), dtype = tf.float64) 
            W2_end = tf.get_variable('W2_end', (self.batch_size, self.batch_size), initializer = tf.constant_initializer(np.eye(self.batch_size)), dtype = tf.float64)
            vt_end = tf.get_variable('vt_end', (self.hidden_size * 4,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)

            W3 = tf.get_variable('W3', (self.batch_size, self.batch_size), initializer=tf.constant_initializer(np.eye(self.batch_size)), dtype = tf.float64)
            # b2_1 = tf.get_variable('b2_1', (self.bill_length,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
            # b2_2 = tf.get_variable('b2_2', (self.bill_length,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)            
            
            # tf.summary.histogram('W1_start', W1_start)
            # tf.summary.histogram('W2_start', W1_start)
            # tf.summary.histogram('W1_end', W1_start)
            # tf.summary.histogram('W2_end', W1_start)
            # tf.summary.histogram('vt_start', W1_start)
            # tf.summary.histogram('vt_end', W1_start)

            for time_step in xrange(self.bill_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
               
                o_t, h_t = cell(bill_embeddings[:, time_step, :], state) # o_t is batch_size, 1
                
                x_start = tf.matmul(W1_start, complete_hidden_states[:, time_step, :]) # result is 1 , hidden_size*4
                y_start = tf.matmul(W2_start, o_t) # result is 1 , hidden_size*4
                u_start = tf.nn.tanh(x_start + y_start) #(batch_size, hidden_size * 4)
                p_start = tf.matmul(u_start, vt_start) #(batch_size, bill_length)
                # p_start = tf.nn.dropout(p_start,self.dropout)

                x_end = tf.matmul(W1_end, complete_hidden_states[:, time_step, :]) # result is 1 , hidden_size*4
                y_end = tf.matmul(W2_end, o_t) # result is 1 , hidden_size*4
                u_end = tf.nn.tanh(x_end + y_end) #(batch_size, hidden_size * 4)
                p_end = tf.matmul(u_end, vt_end) #(batch_size, bill_length)
                #p_end = tf.add(tf.matmul(W3, p_start), p_end)
                #tf.summary.histogram('p_end', p_end)
                # print "preds:"
                # print p_start
                # p_end = tf.nn.dropout(p_end, self.dropout)
                

                p_start = tf.squeeze(p_start)
                p_end = tf.squeeze(p_end)
                preds_start.append(p_start)
                preds_end.append(p_end)
            tf.get_variable_scope().reuse_variables() # set here for each of the next epochs //not working
            assert tf.get_variable_scope().reuse == True
        # return (preds_start, preds_end)       
        preds_start = tf.pack(preds_start)
        # preds_start = tf.squeeze(preds_start)
        preds_start = tf.transpose(preds_start,[1,0])
        preds_end = tf.pack(preds_end)
        # preds_end = tf.squeeze(preds_end)
        preds_end = tf.transpose(preds_end,[1,0])
        # preds_start = tf.nn.softmax(preds_start)
        # preds_end = tf.nn.softmax(preds_end)
        self.predictions = (preds_start, preds_end)
        # tf.summary.histogram('start_preds', self.predictions[0])
        # tf.summary.histogram('end_preds', self.predictions[1])
        # print (preds_start, preds_end)
        return (preds_start, preds_end)

    def add_loss_op(self, preds):
        loss_1 = tf.nn.softmax_cross_entropy_with_logits(preds[0], self.start_index_labels_placeholder)
        loss_2 = tf.nn.softmax_cross_entropy_with_logits(preds[1], self.end_index_labels_placeholder)
        # print 
        # loss_1 = tf.nn.l2_loss(preds[0] - self.start_index_labels_placeholder)
        # loss_2 = tf.nn.l2_loss(preds[1] - self.end_index_labels_placeholder)
        # masked_loss = tf.boolean_mask(loss_1, self.mask_placeholder)
        # tf.summary.histogram('start_preds', preds[0])
        # tf.summary.histogram('end_preds', preds[1])
        # tf.summary.scalar('loss',self.loss)  

        loss = loss_1 + loss_2
        self.loss = tf.reduce_mean(loss)   
        return self.loss

    def add_optimization(self, losses):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)

        grads = [x[0] for x in optimizer.compute_gradients(losses)]
        self.train_op = optimizer.minimize(losses)
        # for grad in grads:
        #     tf.summary.histogram('gradients', grad)
        return self.train_op    


    def test_output(self, sess):
        start_preds = []
        end_preds = []
        for inputs in self.batch_gen_test(self.embedding_wrapper, self.test_data_file):
            start_, end_ = self.predict_on_batch(sess, inputs)
            print "start:"
            print start_
            print "end: "
            print end_
            start_preds += start_.tolist()
            end_preds += end_.tolist()
        return zip(start_preds, end_preds)



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

    def evaluate_helper(self, start_preds, end):
        return 0





    def evaluate_two_hots(self, sess, data_file, indices_file, is_test):
        correct_preds, total_correct, total_preds, number_indices = 0., 0., 0., 0.
        start_num_exact_correct, end_num_exact_correct = 0, 0
        gold_standard_summaries = open(data_file, 'r')
        gold_indices = open(indices_file, 'r')
        file_name = train_name + "/" + str(time.time()) + ".txt"
        if is_test:
            file_name = 'TEST_RESULTS_' + train_name + "/" + str(time.time()) + ".txt"
        
        with open(file_name, 'a') as f:
            for start_preds, end_preds in self.output(sess):
                print "start preds: "
                print start_preds
                print "end preds: "
                print end_preds
                gold = gold_indices.readline()
                gold = gold.split()
                gold_start = int(gold[0])
                gold_end = int(gold[1])

                np_start_preds = np.asarray(start_preds)
                start_maxima = argrelextrema(np_start_preds, np.greater)[0]
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
    
    def predict_on_batch(self, sess, inputs_batch, start_index_labels=None, end_index_labels=None, mask_batch=None, sequence_batch=None, keywords_batch=None):
        feed = self.create_feed_dict(inputs_batch = inputs_batch, start_labels_batch=start_index_labels, masks_batch=mask_batch, sequences = sequence_batch, keywords_batch = None, end_labels_batch = end_index_labels)
        predictions = sess.run(self.predictions, feed_dict=feed)
        # print predictions
        return predictions

    def train_on_batch(self, sess, inputs_batch, start_labels_batch, end_labels_batch, mask_batch, sequence_batch, keywords_batch):
        #print start_labels_batch
        feed = self.create_feed_dict(inputs_batch = inputs_batch, start_labels_batch=start_labels_batch, masks_batch=mask_batch, sequences = sequence_batch, keywords_batch = None, end_labels_batch = end_labels_batch)
        ##### THIS IS SO CONFUSING ######
        _, loss= sess.run([self.train_op, self.loss], feed_dict=feed)
        
        return loss

    def run_epoch(self, sess):
        prog = Progbar(target=1 + int(self.train_len / self.batch_size))
        count = 0
        for inputs,start_labels, end_labels, masks, sequences, keywords in self.batch_generator(self.embedding_wrapper, self.train_data_file, self.train_indices_data_file, self.train_sequence_data_file, self.train_keyword_data_file, self.batch_size, self.bill_length):
            tf.get_variable_scope().reuse_variables()
            loss = self.train_on_batch(sess, inputs, start_labels, end_labels, masks, sequences, keywords)
            prog.update(count + 1, [("train loss", loss)])
            #self.writer.add_summary(summary, count)
            count += 1
        print("")

        print("Evaluating on development data")
        exact_match, entity_scores = self.evaluate_two_hots(sess, self.dev_data_file, self.dev_indices_data_file, False)
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
                    print('New best score! Saving model in /data/'+ train_name+ '/weights/summarizer.weights')
                    saver.save(sess, './data/'+ train_name+ '/weights/summarizer.weights')
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
        model.summary_op = tf.summary.merge_all()
        logger.info("took %.2f seconds", time.time() - start)
        model.writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        tf.get_variable_scope().reuse_variables()
        init = tf.global_variables_initializer()
        tf.get_variable_scope().reuse_variables()
        if not os.path.exists('./data/'+ train_name+ '/weights/'):
            os.makedirs('./data/'+ train_name+ '/weights/')        
        saver = tf.train.Saver()
        
        with tf.Session() as session:
            session.run(init)
            if train:
                model.fit(session, saver)
            
            else:
                print "Testing..."
                print 'Restoring the best model weights found on the dev set'
                saver.restore(session, './data/'+ train_name + '/weights/summarizer.weights')

                # for start_preds, end_preds in model.test_output(session):
                #     print "start preds: "
                #     print start_preds
                #     print "end preds: "
                #     print end_preds


def main():
    mydir = os.path.join(os.getcwd(), train_name)
    os.makedirs(mydir)

    embedding_wrapper = EmbeddingWrapper()
    embedding_wrapper.build_vocab()
    embedding_wrapper.process_glove()
    build_model(embedding_wrapper)
    # test_model()



if __name__ == "__main__":
    main()


