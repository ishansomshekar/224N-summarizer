from embedding_wrapper import EmbeddingWrapper
from read_in_datafile import file_generator
import os
from batch_generator import batch_generator
import numpy as np
from encoder_decoder_cells import DecoderCell
import heapq
import logging
import time

import tensorflow as tf

logger = logging.getLogger("hw3.q2")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class SequencePredictor():
    def __init__(self, embedding_wrapper):
        self.glove_dim = 50
        self.num_epochs = 10
        self.bill_length = 30
        self.lr = 0.05
        self.inputs_placeholder = None
        self.summary_input = None
        self.mask_placeholder = None
        self.hidden_size = 10
        self.predictions = []
        self.batch_size = 1
        self.model_output = os.getcwd() + "model.weights"
        self.train_op = None
        self.loss = 0

        self.start_index_labels_placeholder = None
        self.end_index_labels_placeholder = None
        self.embedding_wrapper = embedding_wrapper
        self.vocab_size = embedding_wrapper.num_tokens
        self.embedding_init = None

        self.train_data_file = "bills_data_100_test.txt"
        self.train_summary_data_file = "extracted_data_full.txt"
        self.train_indices_data_file = "indices_data_100_test.txt"
        self.train_sequence_data_file = "sequence_lengths.txt"

        self.dev_data_file =  "dev_bill_data_100.txt"
        self.dev_summary_data_file =  "extracted_data_full.txt"
        self.dev_indices_data_file = "indices_dev_100.txt"
        self.dev_sequence_data_file = "sequence_len_dev_100.txt"

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
        self.start_index_labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.bill_length))
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

        U_2 = tf.get_variable('U_2', (self.hidden_size * self.bill_length, self.bill_length), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
        b2_2 = tf.get_variable('b2_2', (self.bill_length), \
        initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
        U_3 = tf.get_variable('U_3', (self.bill_length, self.bill_length), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)

        with tf.variable_scope("decoder"):
            dec_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            preds, state = tf.nn.dynamic_rnn(dec_cell,bill_embeddings, sequence_length = self.sequences_placeholder, dtype = tf.float64, initial_state = state)
        preds = tf.reshape(preds, (-1, self.hidden_size * self.bill_length))
        #learn weights for initial index
        preds_start = tf.matmul(preds, U_1) + b2_1
        #learn weights for end index
        #also multiply the start index
        preds_end = tf.matmul(preds, U_2) + tf.matmul(preds_start, U_3) + b2_2
        #self.predictions = (preds_start, preds_end)
        self.predictions = (preds_start, preds_end)
        return (preds_start, preds_end)
        #return (preds_start, preds_end)

    def add_loss_op(self, preds):
        # start_pred = preds[0]
        # end_pred = preds[1]
        start_pred = preds[0]
        end_pred = preds[1]
        loss_1 = tf.nn.softmax_cross_entropy_with_logits(start_pred, self.start_index_labels_placeholder)
        loss_2 = tf.nn.softmax_cross_entropy_with_logits(end_pred, self.end_index_labels_placeholder)
        loss_1 = tf.reduce_mean(loss_1)
        loss_2 = tf.reduce_mean(loss_2)
        loss = loss_1 + loss_2

        # loss = tf.nn.softmax_cross_entropy_with_logits(preds, self.labels_placeholder)
        # loss = tf.reduce_mean(loss)
        self.loss = loss        
        return loss

    def add_optimization(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        train_op = optimizer.minimize(loss)  
        self.train_op = train_op 
        return train_op     

    def output(self, sess):
        batch_preds = []
        for inputs,start_index_labels,end_index_labels, masks, sequences in batch_generator(self.embedding_wrapper, self.dev_data_file, self.dev_indices_data_file, self.dev_sequence_data_file, self.batch_size, self.bill_length):
            preds_ = self.predict_on_batch(sess, inputs, start_index_labels, end_index_labels, masks, sequences)
            batch_preds.append(list(preds_))
        return batch_preds

    def evaluate(self, sess):
        correct_preds, total_correct, total_preds = 0., 0., 0.
        gold_standard = open(self.dev_indices_data_file, 'r')
        batch_preds = self.output(sess)
        for batch_preds in self.output(sess):
            start_index_prediction = batch_preds[0]
            end_index_prediction = batch_preds[1]
            gold = gold_standard.readline()
            gold = gold.split()
            gold_start = int(gold[0])
            gold_end = int(gold[1])
            golds = set()
            golds.add(gold_end)
            golds.add(gold_start)

            # print start_index_prediction
            # print end_index_prediction
            # print
            index_max1 = np.argmax(start_index_prediction)
            index_max2 = np.argmax(end_index_prediction)
            # index_max2 = np.argsort(batch_preds)[-2:]
            prediction = set()
            prediction.add(index_max1)
            prediction.add(index_max2)
            
            correct_preds += len(golds.intersection(prediction))
            total_preds += len(prediction)
            total_correct += len(golds)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

        gold_standard.close()
        return (p, r, f1)
    
    def predict_on_batch(self, sess, inputs_batch, start_index_labels, end_index_labels, mask_batch, sequence_batch):
        feed = self.create_feed_dict(inputs_batch = inputs_batch, start_labels_batch=start_index_labels, end_labels_batch = end_index_labels, masks_batch=mask_batch, sequences = sequence_batch)
        predictions = sess.run(self.predictions, feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, start_labels_batch, end_labels_batch, mask_batch, sequence_batch):
        feed = self.create_feed_dict(inputs_batch = inputs_batch, start_labels_batch=start_labels_batch, end_labels_batch = end_labels_batch, masks_batch=mask_batch, sequences = sequence_batch)
        ##### THIS IS SO CONFUSING ######
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, sess):
        for inputs,start_labels, end_labels, masks, sequences in batch_generator(self.embedding_wrapper, self.train_data_file, self.train_indices_data_file, self.train_sequence_data_file, self.batch_size, self.bill_length):
            loss = self.train_on_batch(sess, inputs, start_labels, end_labels, masks, sequences)
        print("")

        logger.info("Evaluating on development data")
        entity_scores = self.evaluate(sess)
        logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

        f1 = entity_scores[-1]
        return f1

    def fit(self, sess, saver):
        best_score = 0.
        for epoch in range(self.num_epochs):
            logger.info("Epoch %d out of %d", epoch + 1, self.num_epochs)
            score = self.run_epoch(sess)
            if score > best_score:
                best_score = score
                if saver:
                    logger.info("New best score! Saving model in %s", self.model_output)
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


