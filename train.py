from embedding_wrapper import EmbeddingWrapper
from read_in_datafile import file_generator
import os
from batch_generator import batch_generator
import numpy as np

import tensorflow as tf
import seq2seq_model


bill_data_path = 'bill_data_100.txt'
summary_data_path = 'summary_data_100.txt'
all_bill_directory = '/TEST_BILLS/'
all_summary_directory = '/TEST_GOLD_SUMMARIES/'
BATCH_SIZE = 1


class SequencePredictor():
    def __init__(self, num_epochs, glove_dim, embedding_wrapper):
        self.glove_dim = glove_dim
        self.num_epochs = num_epochs
        self.bill_length = 10
        self.summ_length = 10
        self.lr = 0.05
        self.bill_input = None
        self.summary_input = None
        self.embedding_wrapper = embedding_wrapper
        self.vocab_size = embedding_wrapper.num_tokens
        self.embedding_init = None
        self.hidden_size = 500
        self.buckets = [(10, 10)]  
        self.num_layers = 1
        self.max_gradient_norm = 5.0
        self.learning_rate_decay_factor = 0.99

    def create_model(self,session, forward_only):
        """Create model and initialize or load parameters"""
        model = seq2seq_model.Seq2SeqModel(self.vocab_size, self.vocab_size, self.buckets, self.hidden_size, self.num_layers, self.max_gradient_norm, BATCH_SIZE, self.lr, self.learning_rate_decay_factor, forward_only=forward_only)

        # ckpt = tf.train.get_checkpoint_state(gConfig['working_directory'])
        # if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        # print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        # model.saver.restore(session, ckpt.model_checkpoint_path)
        # else:
        # print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
        return model


    def create_feed_dict(self, inputs_batch, embedding_value, labels_batch=None):
        """Creates the feed_dict for the model.
        NOTE: You do not have to do anything here.
        """
        feed_dict = {
            self.bill_input: inputs_batch,
            # self.embedding_init : embedding_value
            }
        if labels_batch is not None:
            feed_dict[self.summary_input] = labels_batch
        return feed_dict

    def fit_model(self):
        #might not need the graph
        with tf.Graph().as_default():
            self.bill_input = tf.placeholder(tf.int32, shape=(None, self.bill_length, self.vocab_size))
            self.summary_input = tf.placeholder(tf.int32, shape=(None, self.summ_length, self.vocab_size))
            
            #  Weights are not the weights that we train on, they are weights given to each word in the sentence
            #  This can be changed to account for masking (set 0 weight to 0-padding) and for attention
            

            #embeddings = tf.Variable(np.load('trimmed_glove.6B.50d.npz'))
            data = np.load('trimmed_glove.6B.50d.npz')
            embeddings = tf.Variable(data['glove'], dtype = tf.float32)
            # embedding_value = data['embeddings']

            print self.bill_input
            bill_embeddings = tf.nn.embedding_lookup(embeddings, self.bill_input)
            print self.bill_length, self.glove_dim
            bill_embeddings = tf.reshape(bill_embeddings, (-1, self.bill_length, self.glove_dim * self.bill_length))        
            summ_embeddings = tf.nn.embedding_lookup(embeddings, self.summary_input)
            summ_embeddings = tf.reshape(summ_embeddings, (-1, self.summ_length, self.glove_dim * self.summ_length))

            #cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size)
            cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
            print summ_embeddings.get_shape()
            print bill_embeddings.get_shape()

            bill_tensor_list = [bill_embeddings[:, i, :] for i in xrange(self.bill_length)]
            summ_tensor_list = [summ_embeddings[:, i, :] for i in xrange(self.summ_length)]

            print summ_tensor_list[0].get_shape()
            #dec_outputs, states = tf.nn.seq2seq.basic_rnn_seq2seq(bill_tensor_list, summ_tensor_list, cell)
            # dec_outputs, states = tf.nn.seq2seq.embedding_rnn_seq2seq(bill_tensor_list, summ_tensor_list, cell, self.vocab_size, self.vocab_size, self.glove_dim)
            #print dec_outputs
            #dec_outputs = tf.pack(dec_outputs)
            #print dec_outputs
            weights = tf.get_variable(name = 'weights', shape = (self.summ_length), initializer=tf.constant_initializer(1))
            # loss = tf.nn.seq2seq.sequence_loss(dec_outputs, summ_tensor_list, weights, self.vocab_size)
            # optimizer = tf.train.AdamOptimizer(self.lr)
            # train_op = optimizer.minimize(loss)

            epoch_losses = []
            with tf.Session() as sess:
                model = self.create_model(sess, False) #sess.run(tf.initialize_all_variables())
                step_time, loss = 0.0, 0.0
                current_step = 0
                previous_losses = []

                for i in xrange(self.num_epochs):
                    batch_losses = []
                    for batch in batch_generator(self.embedding_wrapper, bill_data_path, summary_data_path, BATCH_SIZE, self.bill_length, self.summ_length):
                        for padded_bill, padded_summary in batch:
                            feed = self.create_feed_dict(padded_bill, padded_summary)
                            # train, loss = sess.run([train_op, loss], feed_dict = feed)
                            _, eval_loss, _ = model.step(sess, bill_tensor_list, summ_tensor_list, weights, 0, True)
                            loss += step_loss

                        batch_losses.append(loss)
                    epoch_losses.append(batch_losses)
           
            return epoch_losses

def main():
    bills_datapath = os.getcwd() + all_bill_directory
    gold_summaries_datapath = os.getcwd() + all_summary_directory

    embedding_wrapper = EmbeddingWrapper(bill_data_path)
    embedding_wrapper.build_vocab()
    embedding_wrapper.process_glove()
    # dict_obj = pickle.load(open('vocab.dat', 'r'))

    model = SequencePredictor(1, 50, embedding_wrapper)
    losses = model.fit_model()
    print losses

if __name__ == "__main__":
    main()



