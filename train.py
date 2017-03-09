from embedding_wrapper import EmbeddingWrapper
from read_in_datafile import file_generator
import os
from batch_generator import batch_generator
import numpy as np
from encoder_decoder_cells import DecoderCell
import heapq
import logging
import time
from attention_decoder import attention_decoder
from evaluate_prediction import normalize_answer

from util import Progbar

import tensorflow as tf

ATTENTION_FLAG = 1
UNIDIRECTIONAL_FLAG = True

save_results = False

logger = logging.getLogger("hw3.q2")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

##EDIT THIS
train_name = '3.8 7:44pm'

class SequencePredictor():
    def __init__(self, embedding_wrapper):

        self.glove_dim = 200
        self.num_epochs = 10
        self.bill_length = 500
        self.keywords_length = 5
        self.lr = 0.0005
        self.inputs_placeholder = None
        self.summary_input = None
        self.mask_placeholder = None
        self.hidden_size = 20
        self.predictions = []
        self.batch_size = 50
        self.model_output = os.getcwd() + "model.weights"
        self.train_op = None
        self.loss = 0

        self.start_index_labels_placeholder = None
        self.end_index_labels_placeholder = None
        self.keywords_placeholder = None
        self.embedding_wrapper = embedding_wrapper
        self.vocab_size = embedding_wrapper.num_tokens
        self.embedding_init = None

        self.train_data_file = "train_bills_3_context.txt"
        self.train_summary_data_file = "train_bills_3_summaries.txt"
        self.train_indices_data_file = "train_bills_3_indices.txt"
        self.train_sequence_data_file = "train_bills_3_sequences.txt"
        self.train_keyword_data_file = "train_bills_3_keywords.txt"
        file_open = open(self.train_data_file, 'r')
        self.train_len = len(file_open.read().split("\n"))
        file_open.close()

        self.dev_data_file =  "dev_bills_3_context.txt"
        self.dev_summary_data_file =  "dev_bills_3_summaries.txt"
        self.dev_indices_data_file = "dev_bills_3_indices.txt"
        self.dev_sequence_data_file = "dev_bills_3_sequences.txt"
        self.dev_keyword_data_file = "dev_bills_3_keywords.txt"

        file_open = open(self.dev_data_file, 'r')
        self.dev_len = len(file_open.read().split("\n"))
        file_open.close()

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
        self.start_index_labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.bill_length))
        self.end_index_labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.bill_length))
        self.sequences_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size))
        self.keywords_placeholder = tf.placeholder(tf.int32, shape=(None, self.keywords_length))

    def return_embeddings(self):
        data = np.load('trimmed_glove.6B.200d.npz')
        embeddings = tf.Variable(data['glove'])
        bill_embeddings = tf.nn.embedding_lookup(embeddings, self.inputs_placeholder)
        bill_embeddings = tf.reshape(bill_embeddings, (-1, self.bill_length, self.glove_dim))
        keywords_embeddings = tf.nn.embedding_lookup(embeddings, self.keywords_placeholder)
        keywords_embeddings = tf.reshape(keywords_embeddings, (-1, self.keywords_length, self.glove_dim))
        return bill_embeddings, keywords_embeddings
    
    def attention_prediction_op(self, bill_embeddings, keywords_embeddings):
        with tf.variable_scope("encoder"):
            enc_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            f_outputs, f_state = tf.nn.dynamic_rnn(enc_cell,bill_embeddings, dtype = tf.float64) #outputs is (batch_size, bill_length, hidden_size)
        
        with tf.variable_scope("backwards_encoder"):
            dims = [False, False, True]
            reverse_embeddings = tf.reverse(bill_embeddings, dims) 
            bck_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            b_outputs, b_state = tf.nn.dynamic_rnn(bck_cell,reverse_embeddings, dtype = tf.float64) #outputs is (batch_size, bill_length, hidden_size)
        
        (o_fw, o_bw), _ = tf.nn.bidirectional_dynamic_rnn(enc_cell, dec_cell, bill_embeddings, sequences_placeholder = 5)
        output = tf.concat([f_outputs, f_state], 2)


        with tf.variable_scope('query'):
            fw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            bw_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)

            (o_fw, o_bw), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, keywords_embeddings, sequences_placeholder = 5)
            keyword = tf.concat([o_fw, o_bw], 2)



        enc_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        dec_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        _ , (o_fw, o_bw) = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, bill_embeddings, sequences_placeholder)
        output = tf.concat([o_fw, o_bw], 2)

        enc_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size) ## initialize with last state of query
        dec_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
        (o_fw, o_bw), _ = tf.nn.bidirectional_dynamic_rnn(enc_cell, dec_cell, bill_embeddings, sequences_placeholder = 5)


        


        (query_fw, query_bw), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, keywords_embeddings, sequences_placeholder = 5)
        keyword = tf.concat([query_fw, query_bw], 2)


        (o_fw, o_bw), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, keywords_embeddings, sequences_placeholder = 5)
        (o_fw, o_bw), _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, keywords_embeddings, sequences_placeholder = 5)
        initial_state = tf.zeros(shape = (self.batch_size, self.hidden_size))
        outputs, state = attention_decoder(output, initial_state, keywords_embeddings)

        # complete_outputs = tf.concat(2, [f_outputs, b_outputs] ) #h_t is (batch_size, hiddensize *2 )


        ## Now encode the "query", but in this case some bag of words/keywords

        



        with tf.variable_scope('attention_mechanism'):
            W_1 = tf.get_variable('W_1', (self.hidden_size * 2,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
            W_2 = tf.get_variable('W_2', (self.hidden_size * 2,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)

            m_t = tf.nn.tanh(tf.matmul(W_1, complete_outputs) + tf.matmul(W_2, keyword))
            s_t = tf.nn.softmax(m_t)

            








    def add_unidirectional_prediction_op(self, bill_embeddings):          
        #use hidden states in encoder to make a predictions
        with tf.variable_scope("encoder"):
            enc_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            outputs, state = tf.nn.dynamic_rnn(enc_cell,bill_embeddings, dtype = tf.float64) #outputs is (batch_size, bill_length, hidden_size)
        
        with tf.variable_scope("backwards_encoder"):
            dims = [False, False, True]
            reverse_embeddings = tf.reverse(bill_embeddings, dims) 
            bck_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            b_outputs, b_state = tf.nn.dynamic_rnn(bck_cell,reverse_embeddings, dtype = tf.float64) #outputs is (batch_size, bill_length, hidden_size)
        
        complete_outputs = tf.concat(2, [outputs, b_outputs] ) #h_t is (batch_size, hiddensize *2 )

        preds = []
        bills = []
        with tf.variable_scope("decoder"):
            U_1 = tf.get_variable('U_1', (self.hidden_size * 2,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
            b2_1 = tf.get_variable('b2_1', (self.bill_length,1), \
            initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
            for i in xrange(self.batch_size):
                bill = complete_outputs[i, :, :] #bill is bill_length by hidden_size
                result = tf.matmul(bill, U_1) + b2_1
                result = tf.nn.sigmoid(result)
                preds.append(result)
                bills.append(bill)
        preds = tf.pack(preds)
        preds = tf.squeeze(preds)
        preds = tf.nn.sigmoid(preds)
        self.predictions = preds
        return preds

    def add_loss_op(self, preds):
        # print "preds: ", preds
        # print "labels: ", self.start_index_labels_placeholder

        loss_1 = tf.nn.softmax_cross_entropy_with_logits(preds, self.start_index_labels_placeholder)

        # loss_2 tf.nn.softmax_cross_entropy_with_logits(preds, self.end_index_labels_placeholder)
        #masked_loss = tf.boolean_mask(loss_1, self.mask_placeholder)
        loss = loss_1
        self.loss = loss     
        return self.loss

    def add_optimization(self, losses):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(losses)
        return self.train_op    

    def output(self, sess):
        batch_preds = []
        prog = Progbar(target=1 + int(self.dev_len/ self.batch_size))
        count = 0
        for inputs,start_index_labels,end_index_labels, masks, sequences, keywords in batch_generator(self.embedding_wrapper, self.dev_data_file, self.dev_indices_data_file, self.dev_sequence_data_file, self.dev_keyword_data_file, self.batch_size, self.bill_length):
            preds_ = self.predict_on_batch(sess, inputs, start_index_labels, end_index_labels, masks, sequences, keywords)
            batch_preds.append(list(preds_))
            prog.update(count + 1, [])
            count +=1
        return batch_preds

    def evaluate(self, sess):
        correct_preds, total_correct, total_preds, number_indices = 0., 0., 0., 0.
        start_num_exact_correct, end_num_exact_correct = 0, 0
        gold_standard = open(self.dev_indices_data_file, 'r')
        file_dev = open(self.dev_data_file, 'r')
        file_name = train_name + "/" + str(time.time()) + ".txt"
        with open(file_name, 'a') as f:
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
                    index_max2 = index_prediction_copy.index(maxEnd)

                    #switch the orders if necessary
                    start_index = min(index_max2, index_max1)
                    end_index = max(index_max2, index_max1)

                    text = file_dev.readline()
                    summary = ' '.join(text.split()[start_index:end_index])
                    gold_summary = ' '.join(text.split()[gold_start:gold_end])
                    f.write(summary + ' \n')
                    f.write(gold_summary + ' \n')

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
    
    def predict_on_batch(self, sess, inputs_batch, start_index_labels, end_index_labels, mask_batch, sequence_batch, keywords_batch):
        feed = self.create_feed_dict(inputs_batch = inputs_batch, start_labels_batch=start_index_labels, masks_batch=mask_batch, sequences = sequence_batch, keywords_batch = keywords_batch)
        predictions = sess.run(self.predictions, feed_dict=feed)
        return predictions

    def train_on_batch(self, sess, inputs_batch, start_labels_batch, end_labels_batch, mask_batch, sequence_batch, keywords_batch):
        #print start_labels_batch
        feed = self.create_feed_dict(inputs_batch = inputs_batch, start_labels_batch=start_labels_batch, masks_batch=mask_batch, sequences = sequence_batch, keywords_batch = keywords_batch)
        ##### THIS IS SO CONFUSING ######
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def run_epoch(self, sess):
        prog = Progbar(target=1 + int(self.train_len / self.batch_size))
        count = 0
        for inputs,start_labels, end_labels, masks, sequences, keywords in batch_generator(self.embedding_wrapper, self.train_data_file, self.train_indices_data_file, self.train_sequence_data_file, self.train_keyword_data_file, self.batch_size, self.bill_length):
            loss = self.train_on_batch(sess, inputs, start_labels, end_labels, masks, sequences, keywords)
            prog.update(count + 1, [("train loss", max(loss))])
            count += 1
        print("")

        print("Evaluating on development data")
        exact_match, entity_scores = self.evaluate(sess)
        print("Entity level end_exact_match/start_exact_match/P/R/F1: %.2f/%.2f/%.2f/%.2f", exact_match[0], exact_match[1], entity_scores[0], entity_scores[1], entity_scores[2])

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
        bill_embeddings, keywords_embeddings = self.return_embeddings()
        if UNIDIRECTIONAL_FLAG:
            logger.info("Running unidirectional...",)
            preds = self.add_unidirectional_prediction_op(bill_embeddings)
        else:
            logger.info("Running attentive...",)
            preds = self.add_attentive_predictor(bill_embeddings, keywords_embeddings)
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
    if save_results:
        mydir = os.path.join(os.getcwd(), train_name)
        os.makedirs(mydir)

    embedding_wrapper = EmbeddingWrapper()
    embedding_wrapper.build_vocab()
    embedding_wrapper.process_glove()
    build_model(embedding_wrapper)



if __name__ == "__main__":
    main()


