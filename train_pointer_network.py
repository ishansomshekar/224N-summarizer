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
import rougescore as rs
#from pointer_network import PointerCell

from util import Progbar

import tensorflow as tf

ATTENTION_FLAG = 1
UNIDIRECTIONAL_FLAG = True

logger = logging.getLogger("hw3.q2")
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

##EDIT THIS
train_name = '3_11_830am'

class SequencePredictor():
    def __init__(self, embedding_wrapper):

        self.glove_dim = 50
        self.num_epochs = 10
        self.bill_length = 50
        self.keywords_length = 5
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
        
        complete_outputs = tf.concat(2, [outputs, b_outputs]) #h_t is (batch_size, hidden_size *2 )

        preds_start = []
        preds_end = []
        with tf.variable_scope("decoder"):
            U_1_start = tf.get_variable('U_1_start', (self.hidden_size * 2,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
            U_1_end = tf.get_variable('U_1_end', (self.hidden_size * 2,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
            b2_1 = tf.get_variable('b2_1', (self.bill_length,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
            b2_2 = tf.get_variable('b2_2', (self.bill_length,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
            for i in xrange(self.batch_size):
                bill = complete_outputs[i, :, :] #bill is bill_length by hidden_size
                result_start = tf.matmul(bill, U_1_start) + b2_1
                result_end = tf.matmul(bill, U_1_end) + b2_2
                preds_start.append(result_start)
                preds_end.append(result_end)
        preds_start = tf.pack(preds_start)
        preds_end = tf.pack(preds_end)
        preds_start = tf.squeeze(preds_start)
        preds_end = tf.squeeze(preds_end)

        self.predictions = preds_start, preds_end
        return preds_start, preds_end

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

        preds = []
        with tf.variable_scope("decoder"):
            # tf.get_variable_scope().reuse_variables() doesnt work because of first pass through loop
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size*4)
            state = cell.zero_state(self.batch_size, dtype=tf.float64)
            W1 = tf.get_variable('W1', (self.batch_size, self.batch_size), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64) 
            W2 = tf.get_variable('W2', (self.batch_size, self.batch_size), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
            vt = tf.get_variable('vt', (self.hidden_size * 4,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
            # b2_1 = tf.get_variable('b2_1', (self.bill_length,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
            # b2_2 = tf.get_variable('b2_2', (self.bill_length,1), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)            
            for time_step in xrange(self.bill_length):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
               
                o_t, h_t = cell(bill_embeddings[:, time_step, :], state) # o_t is batch_size, 1
                x = tf.matmul(W1, complete_hidden_states[:, time_step, :]) # result is 1 , hidden_size*4
                y = tf.matmul(W2, o_t) # result is 1 , hidden_size*4
                u = tf.nn.tanh(x + y) #(batch_size, hidden_size * 4)
                p = tf.matmul(u, vt) #(batch_size, bill_length)

                preds.append(tf.nn.softmax(p))
            tf.get_variable_scope().reuse_variables() # set here for each of the next epochs //not working
            assert tf.get_variable_scope().reuse == True
        preds = tf.pack(preds)
        preds = tf.squeeze(preds)
        preds = tf.transpose(preds,[1,0])
        self.predictions = preds
        return preds

    def add_loss_op(self, preds):
        # start_indexes = tf.cast(preds[0], tf.float64)
        # end_indexes = tf.cast(preds[1], tf.float64)
        # start_indexes = tf.argmax(start_indexes, 1)
        # end_indexes = tf.argmax(end_indexes, 1)
        # start_indexes = tf.cast(start_indexes, dtype = tf.float64)
        # end_indexes = tf.cast(end_indexes, dtype = tf.float64)

        # label_start_indexes = tf.argmax(self.start_index_labels_placeholder, 1)
        # label_end_indexes = tf.argmax(self.end_index_labels_placeholder, 1)
        # label_start_indexes = tf.cast(label_start_indexes, dtype = tf.float64)
        # label_end_indexes = tf.cast(label_end_indexes, dtype = tf.float64)
        # print label_start_indexes
        # print label_end_indexes

        # print start_indexes
        # print end_indexes

        # start_diff = tf.subtract(label_start_indexes,start_indexes)
        # total_start_loss = tf.nn.l2_loss(start_diff)
        # end_diff = tf.subtract(label_end_indexes,end_indexes)
        # total_end_loss = tf.nn.l2_loss(end_diff)
        # total_start_loss = tf.reduce_mean(total_start_loss)
        # total_end_loss = tf.reduce_mean(total_end_loss)

        # self.loss = total_start_loss + total_end_loss
        # return tf.add(total_start_loss, total_end_loss)


        loss_1 = tf.nn.softmax_cross_entropy_with_logits(preds, self.start_index_labels_placeholder)
        # masked_loss = tf.boolean_mask(loss_1, self.mask_placeholder)
        loss = loss_1
        self.loss = tf.reduce_mean(loss)   
        return self.loss

    def add_optimization(self, losses):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
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
                    index_max2 = index_prediction_copy.index(maxEnd)

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
        for inputs,start_labels, end_labels, masks, sequences, keywords in batch_generator(self.embedding_wrapper, self.train_data_file, self.train_indices_data_file, self.train_sequence_data_file, self.train_keyword_data_file, self.batch_size, self.bill_length):
            tf.get_variable_scope().reuse_variables()
            loss = self.train_on_batch(sess, inputs, start_labels, end_labels, masks, sequences, keywords)
            prog.update(count + 1, [("train loss", loss)])
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
        # if UNIDIRECTIONAL_FLAG:
        #     logger.info("Running unidirectional...",)
        #     preds = self.add_unidirectional_prediction_op(bill_embeddings)
        # else:
        logger.info("Running attentive...",)
        preds = self.add_pointer_prediction_op(bill_embeddings)
        loss = self.add_loss_op(preds)
        self.train_op = self.add_optimization(loss)
        return preds, loss, self.train_op

def build_model(embedding_wrapper):
    with tf.Graph().as_default():
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


