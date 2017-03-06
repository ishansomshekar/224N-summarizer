from embedding_wrapper import EmbeddingWrapper
from read_in_datafile import file_generator
import os
from batch_generator import batch_generator
import numpy as np

import tensorflow as tf

unique_clean_bill_names_85 = 'test_bill_names.csv'
all_bill_directory = '/TEST_BILLS/'
all_summary_directory = '/TEST_GOLD_SUMMARIES/'
BATCH_SIZE = 2


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

        ## BUILD PLACEHOLDERS

        ## BUILD LOSS FUNCTION

        ## BUILD OPTIMIZER

        ## train batch

        #might not need the graph
        with tf.Graph().as_default():
            #
            # self.bill_input = tf.placeholder(tf.float32, shape=(None, self.bill_length))
            # self.summary_input = tf.placeholder(tf.float32, shape=(None, self.summ_length))
            self.bill_input = tf.placeholder(tf.int32, shape=(None, self.bill_length, self.vocab_size))
            self.summary_input = tf.placeholder(tf.int32, shape=(None, self.summ_length, self.vocab_size))
            #self.embedding_init = tf.placeholder(tf.float32, shape = (self.vocab_size, self.glove_dim))
            #  Weights are not the weights that we train on, they are weights given to each word in the sentence
            #  This can be changed to account for masking (set 0 weight to 0-padding) and for attention
            #weights = tf.get_variable(name = 'weights', shape = (self.summ_length), initializer=tf.constant_initializer(0))
            #embeddings = tf.Variable(np.load('trimmed_glove.6B.50d.npz'))
            data = np.load('trimmed_glove.6B.50d.npz')
            embeddings = tf.Variable(data['glove'])
            # embedding_value = data['embeddings']
            # print embeddings.type()
            print self.bill_input
            bill_embeddings = tf.nn.embedding_lookup(embeddings, self.bill_input)
            bill_embeddings = tf.reshape(bill_embeddings, (-1, self.bill_length, self.glove_dim))        
            summ_embeddings = tf.nn.embedding_lookup(embeddings, self.summary_input)
            summ_embeddings = tf.reshape(summ_embeddings, (-1, self.summ_length, self.glove_dim))

            preds = []
            #10 is my dummy variable for hidden size
            enc_cell = tf.nn.rnn_cell.LSTMCell(10)
            state = tf.zeros((BATCH_SIZE, 10), tf.float32)
            outputs, state = tf.nn.dynamic_rnn(enc_cell, bill_embeddings, state)

            dec_cell = tf.nn.rnn_cell.LSTMCell(10)
            dec_state = state
            dec_outputs, dec_state = tf.nn.dynamic_rnn(dec_cell, outputs, dec_state)
            # with tf.variable_scope("RNN"):
            #     for time_step in range(self.bill_length):
            #         if time_step > 0:
            #             tf.get_variable_scope().reuse_variables()
            #         o_t, h_t = cell(x[:, time_step, :], state)
            #         o_drop_t = tf.nn.dropout(o_t, dropout_rate)
            #         pred = tf.matmul(o_drop_t, U) + b2
            #         preds.append(pred)
            #         state = h_t
            #         ### END YOUR CODE 

            dec_outputs, states = tf.seq2seq.embedding_rnn_seq2seq(self.bill_input, self.summary_input, cell, self.vocab_size, self.vocab_size)


            loss = seq2seq.sequence_loss(dec_outputs, self.summary_input, weights, self.vocab_size)
            optimizer = tf.train.AdamOptimizer(self.lr)
            train_op = optimizer.minimize(loss)

            epoch_losses = []
            with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())


            #inserted for testing
                for i in xrange(self.num_epochs):
                    batch_losses = []
                    for batch in batch_generator(self.embedding_wrapper, unique_clean_bill_names_85,BATCH_SIZE, self.bill_length, self.summ_length):
                        for padded_bill, padded_summary in batch:
                            # print padded_bill
                            # print padded_summary
                            # print

                            # do additional preprociessing to change the indexes to one hot vectors inside generate batch
                            feed = self.create_feed_dict(padded_bill, padded_summary)#, embedding_value)
                            train, loss = sess.run([train_op, loss], feed_dict = feed)

                        batch_losses.append(loss)
                    epoch_losses.append(batch_losses)

                # for i in xrange(self.num_epochs):
                    
                #     batch_losses = []
                #     for padded_bill, padded_summary in batch_generator(embedding_wrapper, unique_clean_bill_names_85 ,BATCH_SIZE, MAX_BILL_LENGTH, MAX_SUMMARY_LENGTH):
                #         feed = self.create_feed_dict(padded_bill, padded_summary)#, embedding_value)
                #         train, loss = sess.run([train_op, loss], feed_dict = feed)

                #         batch_losses.append(loss)
                #     epoch_losses.append(batch_losses)
           
            return epoch_losses



            ##initialize variables
            ## loop over epochs
            ## loop over the batches
            # extract batch
            # build feed dictionary
            # loss, grad = sess.run()

        # for every epoch:
        #     for every batch:
        #         get the batch from data
        #         model.step / model. run

        #         pass grad_norm and losses up through
        #         every once in a while run eval on losses


def main():
    bills_datapath = os.getcwd() + all_bill_directory
    gold_summaries_datapath = os.getcwd() + all_summary_directory

    embedding_wrapper = EmbeddingWrapper(bills_datapath)
    embedding_wrapper.build_vocab()
    embedding_wrapper.process_glove()

    model = SequencePredictor(1, 50, embedding_wrapper)
    losses = model.fit_model()
    print losses

if __name__ == "__main__":
    main()


