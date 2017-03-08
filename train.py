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

ATTENTION_FLAG = 1
UNIDIRECTIONAL_FLAG = 0

logger = logging.getLogger("hw3.q2")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class SequencePredictor():
    def __init__(self, embedding_wrapper):
        self.glove_dim = 50
        self.num_epochs = 10
        self.bill_length = 500
        self.lr = 0.0001
        self.inputs_placeholder = None
        self.summary_input = None
        self.mask_placeholder = None
        self.hidden_size = 10
        self.predictions = []
        self.batch_size = 50
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
        b2_1 = tf.get_variable('b2_1', (self.batch_size,self.bill_length), \
        initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)

        U_2 = tf.get_variable('U_2', (self.hidden_size, self.bill_length), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
        b2_2 = tf.get_variable('b2_2', shape = (self.batch_size,self.bill_length), \
        initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)

        with tf.variable_scope("decoder"):
            dec_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            for time_step in range(self.bill_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                o_t, h_t = dec_cell(bill_embeddings[:, time_step, :], state)
                h_t = h_t[0] + h_t[1]
                h_t = tf.matmul(h_t, U_2) #+ b2_2
                preds.append(h_t)
        preds = tf.pack(preds)
        preds = tf.transpose(preds, [1,0,2])     
        self.predictions = preds
        return preds

    def attention(self, h_t, encoder_hs, W_c, b_c):
        #scores = [tf.matmul(tf.tanh(tf.matmul(tf.concat(1, [h_t, tf.squeeze(h_s, [0])]),
        #                    self.W_a) + self.b_a), self.v_a)
        #          for h_s in tf.split(0, self.max_size, encoder_hs)]
        #scores = tf.squeeze(tf.pack(scores), [2])
        scores = tf.reduce_sum(tf.mul(encoder_hs, h_t), 2)
        a_t    = tf.nn.softmax(tf.transpose(scores))
        a_t    = tf.expand_dims(a_t, 2)
        c_t    = tf.batch_matmul(tf.transpose(encoder_hs, perm=[1,2,0]), a_t)
        c_t    = tf.squeeze(c_t, [2])
        h_tld  = tf.tanh(tf.matmul(tf.concat(1, [h_t, c_t]), W_c) + b_c)

        return h_tld

    def add_attentive_predictor(self, bill_embeddings):
        dims = [False, False, True]
        reverse_embeddings = tf.reverse(bill_embeddings, dims)        
        h_outputs = []
        c_state = tf.zeros(shape=(self.batch_size,self.hidden_size), dtype=tf.float64)
        m_state = tf.zeros(shape=(self.batch_size,self.hidden_size), dtype=tf.float64)
        state = (c_state, m_state)
        f_outputs = []
        b_outputs = []
        with tf.variable_scope('encoder'):

            forward_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            for time_step in range(self.bill_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                o_t, h_t = forward_cell(bill_embeddings[:, time_step, :], state)    
                f_outputs.append(o_t)
                state = h_t
            
            backward_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            state = (c_state, m_state)
            for time_step in range(self.bill_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                bo_t, bh_t = backward_cell(reverse_embeddings[:, time_step, :], state)
                b_outputs.append(bo_t)
                state = bh_t

        for i in xrange(len(f_outputs)):
            h_outputs = (tf.concat(2, [f_outputs, b_outputs]))

        W_h_outputs = tf.get_variable('W_h_outputs', shape=(self.bill_length, self.hidden_size * 2, self.hidden_size), initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
        h_outputs = tf.matmul(h_outputs, W_h_outputs)
        h_outputs = tf.pack(h_outputs)


        W = tf.get_variable('W', shape=(self.hidden_size, self.hidden_size), initializer=tf.contrib.layers.xavier_initializer())
        w = tf.get_variable('w', shape=(self.hidden_size), initializer=tf.contrib.layers.xavier_initializer())

        preds = []
        with tf.variable_scope('decoder'):
            t_proj_W = tf.get_variable("t_proj_W", shape=(self.glove_dim, self.hidden_size),
                                            initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64)
            t_proj_b = tf.get_variable("t_proj_b", shape=(self.hidden_size),
                                            initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64)
            #projection
            proj_W = tf.get_variable("W", shape=(self.hidden_size, self.glove_dim),
                                          initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64)
            proj_b = tf.get_variable("b", shape=(self.glove_dim),
                                          initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64)
            proj_Wo = tf.get_variable("Wo", shape=(self.glove_dim, self.bill_length),
                                           initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64)
            proj_bo = tf.get_variable("bo", shape=[self.bill_length],
                                           initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64)

            # attention
            #v_a = tf.get_variable("v_a", shape=(self.hidden_size, 1),
                                       initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64)
            # W_a = tf.get_variable("W_a", shape=(2*self.hidden_size, self.hidden_size),
            #                            initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64)
            b_a = tf.get_variable("b_a", shape=(self.hidden_size),
                                       initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64)
            W_c = tf.get_variable("W_c", shape=(2*self.hidden_size, self.hidden_size),
                                       initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64)
            b_c = tf.get_variable("b_c", shape=(self.hidden_size),
                                       initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64)

            c_state = tf.zeros(shape=(self.batch_size,self.hidden_size), dtype=tf.float64)
            m_state = tf.zeros(shape=(self.batch_size,self.hidden_size), dtype=tf.float64)
            s = (c_state, m_state)

            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            for time_step in range(self.bill_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                x = bill_embeddings[:, time_step, :]
                x = tf.matmul(x, t_proj_W) + t_proj_b
                o_t, h_t = cell(x, s)
                h_t = s
                h_tld = self.attention(o_t, h_outputs, W_c, b_c)

                oemb  = tf.matmul(h_tld, proj_W) + proj_b
                logit = tf.matmul(oemb, proj_Wo) + proj_bo
                preds.append(logit)

        self.predictions = preds
        return preds

    def add_loss_op(self, preds):
        start_pred = preds
        loss_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(start_pred, self.start_index_labels_placeholder)
        masked_loss = tf.boolean_mask(loss_1, self.mask_placeholder)
        loss = tf.reduce_mean(loss_1) 
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
        file_dev = open(self.dev_data_file, 'r')
        file_name = 'model_results' + str(time.time()) + ".txt"
        with open(file_name, 'a') as f:
            for batch_preds in self.output(sess):
                start_index_prediction = batch_preds[0]
                #end_index_prediction = batch_preds[1]
                gold = gold_standard.readline()
                gold = gold.split()
                gold_start = int(gold[0])

                start_index_prediction = start_index_prediction.tolist()
                #end_index_prediction = end_index_prediction.tolist()
                maxStart = max(start_index_prediction)
                #maxEnd = max(end_index_prediction)
                index_max1 = start_index_prediction.index(maxStart)
                #index_max2 = end_index_prediction.index(maxEnd)
                text = file_dev.readline()
                summary = ' '.join(text.split()[index_max1:index_max1 +20])
                gold_summary = ' '.join(text.split()[gold_start:gold_start+20])
                f.write('our summary: ' + summary + ' \n')
                f.write('gold summary: ' + gold_summary + ' \n')

                if index_max1 == gold_start:
                    correct_preds += 1
                total_preds += 1
                total_correct += 1

            p = correct_preds / total_preds if correct_preds > 0 else 0
            r = correct_preds / total_correct if correct_preds > 0 else 0
            f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0

            gold_standard.close()


            f.write('Model results: \n')
            f.write('learning rate: %d \n' % self.lr)
            f.write('batch size: %d \n' % self.batch_size)
            f.write('hidden size: %d \n' % self.hidden_size)
            f.write('bill_length: %d \n' % self.bill_length)
            f.write('bill_file: %s \n' % self.train_data_file)
            f.write('dev_file: %s \n' % self.dev_data_file)
            f.write("Epoch P/R/F1: %.2f/%.2f/%.2f \n" % (p, r, f1))
            f.close()
        
        return (p, r, f1)
    
    def predict_on_batch(self, sess, inputs_batch, start_index_labels, end_index_labels, mask_batch, sequence_batch):
        feed = self.create_feed_dict(inputs_batch = inputs_batch, start_labels_batch=start_index_labels, masks_batch=mask_batch, sequences = sequence_batch)
        predictions = sess.run(tf.argmax(self.predictions,axis = 2), feed_dict=feed)
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
        epoch_scores = []
        for epoch in range(self.num_epochs):
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
        bill_embeddings = self.return_embeddings()
        if UNIDIRECTIONAL_FLAG:
            preds = self.add_unidirectional_prediction_op(bill_embeddings)
        else:
            preds = self.add_attentive_predictor(bill_embeddings)
        loss = self.add_loss_op(preds)
        self.train_op = self.add_optimization(loss)
        return preds, loss, self.train_op

def build_model(embedding_wrapper):
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
    build_model(embedding_wrapper)



if __name__ == "__main__":
    main()


