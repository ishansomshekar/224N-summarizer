from embedding_wrapper import EmbeddingWrapper
from read_in_datafile import file_generator
import os
from batch_generator import batch_generator
import numpy as np
from encoder_decoder_cells import DecoderCell
import heapq
import logging
import time

from util import Progbar

import tensorflow as tf

logger = logging.getLogger("hw3.q2")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class SequencePredictor():
    def __init__(self, embedding_wrapper):
        self.glove_dim = 50
        self.num_epochs = 10
        self.bill_length = 200
        self.lr = 0.0001
        self.inputs_placeholder = None
        self.summary_input = None
        self.mask_placeholder = None
        self.hidden_size = 10
        self.predictions = []
        self.batch_size = 5
        self.model_output = os.getcwd() + "model.weights"
        self.train_op = None
        self.loss = 0

        self.start_index_labels_placeholder = None
        self.end_index_labels_placeholder = None
        self.embedding_wrapper = embedding_wrapper
        self.vocab_size = embedding_wrapper.num_tokens
        self.embedding_init = None

        self.train_data_file = "train_data_extracted_full.txt"
        self.train_summary_data_file = "extracted_data_full.txt"
        self.train_indices_data_file = "train_indices_data_full.txt"
        self.train_sequence_data_file = "train_sequence_lengths.txt"
        file_open = open(self.train_data_file, 'r')
        self.train_len = len(file_open.read().split("\n"))
        file_open.close()

        self.dev_data_file =  "dev_data_extracted_full.txt"
        self.dev_summary_data_file =  "extracted_data_full.txt"
        self.dev_indices_data_file = "dev_indices_data_full.txt"
        self.dev_sequence_data_file = "dev_sequence_lengths.txt"

        file_open = open(self.dev_data_file, 'r')
        self.dev_len = len(file_open.read().split("\n"))
        file_open.close()

    def create_feed_dict(self, inputs_batch, masks_batch, sequences, start_labels_batch = None, end_labels_batch = None):
        feed_dict = {
            self.inputs_placeholder : inputs_batch,
            self.mask_placeholder : masks_batch,
            self.sequences_placeholder : sequences
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
        self.end_index_labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.bill_length))
        self.sequences_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size))

    def return_embeddings(self):
        data = np.load('trimmed_glove.6B.50d.npz')
        embeddings = tf.Variable(data['glove'])
        bill_embeddings = tf.nn.embedding_lookup(embeddings, self.inputs_placeholder)
        bill_embeddings = tf.reshape(bill_embeddings, (-1, self.bill_length, self.glove_dim))
        return bill_embeddings

    def add_unidirectional_prediction_op(self, bill_embeddings):          
        with tf.variable_scope("encoder"):
            enc_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            outputs, state = tf.nn.dynamic_rnn(enc_cell,bill_embeddings, dtype = tf.float64)
        #decoder
        preds = []
        U_1 = tf.get_variable('U_1', (self.hidden_size * self.bill_length, self.bill_length), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
        b2_1 = tf.get_variable('b2_1', (self.bill_length), \
        initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)

        U_2 = tf.get_variable('U_2', (self.hidden_size, 1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
        b2_2 = tf.get_variable('b2_2', shape = (), \
        initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)

        with tf.variable_scope("decoder"):
            dec_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            for time_step in range(self.bill_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                o_t, h_t = dec_cell(bill_embeddings[:, time_step, :], state)
                h_t = h_t[0] + h_t[1]
                h_t = tf.matmul(h_t, U_2) + b2_2
                preds.append(h_t)
                
        preds = tf.nn.sigmoid(preds)
        preds = tf.reshape(preds, (self.batch_size, self.bill_length))
        self.predictions = preds
        return preds

    def add_loss_op(self, preds):
        start_pred = preds
        loss_1 = tf.nn.softmax_cross_entropy_with_logits(start_pred, self.start_index_labels_placeholder)
        loss_1 = tf.reduce_mean(loss_1)
        loss = loss_1 
        self.loss = loss        
        return loss

    def add_optimization(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        train_op = optimizer.minimize(loss)  
        self.train_op = train_op 
        return train_op     

    def output(self, sess):
        batch_preds = []
        prog = Progbar(target=1 + int(self.dev_len/ self.batch_size))
        count = 0
        for inputs,start_index_labels,end_index_labels, masks, sequences in batch_generator(self.embedding_wrapper, self.dev_data_file, self.dev_indices_data_file, self.dev_sequence_data_file, self.batch_size, self.bill_length):
            preds_ = self.predict_on_batch(sess, inputs, start_index_labels, end_index_labels, masks, sequences)
            batch_preds.append(list(preds_))
            prog.update(count + 1, [])
            count +=1
        return batch_preds

    def evaluate(self, sess):
        correct_preds, total_correct, total_preds = 0., 0., 0.
        gold_standard = open(self.dev_indices_data_file, 'r')
        batch_preds = self.output(sess)
        file_dev = open(self.dev_data_file, 'r')
        for batch_preds in self.output(sess):
            start_index_prediction = batch_preds[0]
            #end_index_prediction = batch_preds[1]
            gold = gold_standard.readline()
            gold = gold.split()
            gold_start = int(gold[0])
            #gold_end = int(gold[1])
            #golds = set()
            #golds.add(gold_end)
            #golds.add(gold_start)

            start_index_prediction = start_index_prediction.tolist()
            #end_index_prediction = end_index_prediction.tolist()
            maxStart = max(start_index_prediction)
            #maxEnd = max(end_index_prediction)
            index_max1 = start_index_prediction.index(maxStart)
            #index_max2 = end_index_prediction.index(maxEnd)
            print index_max1
            print gold_start
            print
            #prediction = set()
            #prediction.add(index_max1)
            #prediction.add(index_max2)

            text = file_dev.readline()
            #first_index = min(index_max1, index_max2)
            #sec_index = max(index_max1, index_max2)
            if index_max1 == gold_start:
                correct_preds += 1
            total_preds += 1
            total_correct += 1

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

        gold_standard.close()
        return (p, r, f1)
    
    def predict_on_batch(self, sess, inputs_batch, start_index_labels, end_index_labels, mask_batch, sequence_batch):
        feed = self.create_feed_dict(inputs_batch = inputs_batch, start_labels_batch=start_index_labels, masks_batch=mask_batch, sequences = sequence_batch)
        predictions = sess.run(self.predictions, feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, start_labels_batch, end_labels_batch, mask_batch, sequence_batch):
        feed = self.create_feed_dict(inputs_batch = inputs_batch, start_labels_batch=start_labels_batch, masks_batch=mask_batch, sequences = sequence_batch)
        ##### THIS IS SO CONFUSING ######
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
	return loss

    def run_epoch(self, sess):
        prog = Progbar(target=1 + int(self.train_len / self.batch_size))
        count = 0
        for inputs,start_labels, end_labels, masks, sequences in batch_generator(self.embedding_wrapper, self.train_data_file, self.train_indices_data_file, self.train_sequence_data_file, self.batch_size, self.bill_length):
            loss = self.train_on_batch(sess, inputs, start_labels, end_labels, masks, sequences)
            prog.update(count + 1, [("train loss", loss)])
            count += 1
        print("")

        print("Evaluating on development data")
        entity_scores = self.evaluate(sess)
        print("Entity level P/R/F1: %.2f/%.2f/%.2f", entity_scores[0], entity_scores[1], entity_scores[2])

        f1 = entity_scores[-1]
        return f1

    def fit(self, sess, saver):
        best_score = 0.
        for epoch in range(self.num_epochs):
            print("Epoch %d out of %d", epoch + 1, self.num_epochs)
            score = self.run_epoch(sess)
            if score > best_score:
                best_score = score
                if saver:
                    print("New best score! Saving model in %s", self.model_output)
                    saver.save(sess, self.model_output)
            print("")
        return best_score

    def initialize_model(self):
        self.add_placeholders()
        bill_embeddings = self.return_embeddings()
        preds = self.add_unidirectional_prediction_op(bill_embeddings)
        loss = self.add_loss_op(preds)
        self.train_op = self.add_optimization(loss)
        return preds, loss, self.train_op

def run_unidirectional(embedding_wrapper):
    with tf.Graph().as_default():
        logger.info("Building model...",)
        start = time.time()
        model = SequencePredictor(embedding_wrapper)
        preds, loss, train_op = model.initialize_model()
        logger.info("took %.2f seconds", time.time() - start)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(init)
            model.fit(session, saver)

def main():
    embedding_wrapper = EmbeddingWrapper()
    embedding_wrapper.build_vocab()
    embedding_wrapper.process_glove()
    run_unidirectional(embedding_wrapper)


if __name__ == "__main__":
    main()


