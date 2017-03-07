from embedding_wrapper import EmbeddingWrapper
from read_in_datafile import file_generator
import os
from batch_generator import batch_generator
import numpy as np
from encoder_decoder_cells import DecoderCell

import tensorflow as tf

bill_data_file = "bill_data_extracted_full.txt"
summary_data_file = "extracted_data_full.txt"
indices_data_file = "indices_data_full.txt"

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
        self.mask_placeholder = None
        self.labels_placeholder = None
        self.embedding_wrapper = embedding_wrapper
        self.vocab_size = embedding_wrapper.num_tokens
        self.embedding_init = None

    def create_feed_dict(self, inputs_batch, labels_batch, mask_batch):
        """Creates the feed_dict for the model.
        NOTE: You do not have to do anything here.
        """
        # print inputs_batch
        # print
        # print labels_batch

        # maskbatch = [True for i in xrange(self.bill_length)]
        # label = [0 for i in xrange(self.bill_length)]
        # label[3] = 1
        feed_dict = {
            self.bill_input : inputs_batch,
            self.mask_placeholder : mask_batch
            # self.embedding_init : embedding_value
            }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
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
            self.mask_placeholder = tf.placeholder(tf.bool, shape=(None, self.bill_length))
            self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.bill_length))
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

            
            #10 is my dummy variable for hidden size
            
            #state = tf.zeros((self.bill_length), dtype = tf.float32)
            with tf.variable_scope("encoder"):
                enc_cell = tf.nn.rnn_cell.LSTMCell(10)
                outputs, state = tf.nn.dynamic_rnn(enc_cell,bill_embeddings, dtype = tf.float64)
            #state = tf.nn.sigmoid(state)
            #outputs, state = tf.nn.dynamic_rnn(enc_cell, bill_embeddings, initial_state = state)
            #dec_cell = tf.nn.rnn_cell.LSTMCell(10)
            #dec_cell = DecoderCell(10)
            #dec_state = state
            #preds = tf.pack(preds)
            #preds = tf.transpose(preds, [1,0,2])
            #outputs = tf.cast(outputs, tf.float32)
            #preds = tf.cast(preds, tf.float32)
            with tf.variable_scope("decoder"):
                dec_cell = tf.nn.rnn_cell.LSTMCell(10)
                dec_outputs, dec_final_state = tf.nn.dynamic_rnn(dec_cell, bill_embeddings, initial_state = state)
            #input_ = dummy_starter
            # with tf.variable_scope("decoder"):
            #     for i in xrange(self.summ_length+1):
            #         W_d_in = tf.get_variable("W_d_in", [input_dimensions, lstm_width], initializer=init)   # S x L
            #         b_d_in = tf.get_variable("b_d_in", [batch_size, lstm_width], initializer=init)         # B x L
            #         cell_input = tf.nn.elu(tf.matmul(input_, W_d_in) + b_d_in)                               # B x L
                    
            #         step_output, dec_state = dec_cell(cell_input, enc_final_state)

                    ## need to find a way to reformat the output into the input
                    ## dimensions need to change from B x L to B x S


                # for time_step in range(self.summ_length):
                #     if time_step > 0:
                #         tf.get_variable_scope().reuse_variables()
                #     o_t, h_t = cell(x[:, time_step, :], state)
                #     o_drop_t = tf.nn.dropout(o_t, dropout_rate)
                #     pred = tf.matmul(o_drop_t, U) + b2
                #     preds.append(pred)
                #     state = h_t

            print dec_outputs
            print "#################"
            print self.labels_placeholder

            softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(dec_outputs, self.labels_placeholder)
            
            masked_loss = tf.boolean_mask(softmax_loss, self.mask_placeholder)
            loss = tf.reduce_mean(masked_loss)


            # loss = seq2seq.sequence_loss(dec_outputs, self.summary_input, weights, self.vocab_size)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            train_op = optimizer.minimize(loss)

            epoch_losses = []
            with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())


            #inserted for testing
                for i in xrange(self.num_epochs):
                    batch_losses = []
                    for batch in batch_generator(self.embedding_wrapper, bill_data_file, indices_data_file, BATCH_SIZE, self.bill_length, self.summ_length):
                        # do additional preprociessing to change the indexes to one hot vectors inside generate batch
                        # print batch
                        # mask_batch = [True for i in xrange(self.bill_length)]
                        # labels = [0 for i in xrange(self.bill_length)]
                        # labels[3] = 1
                        print len(batch[0]), len(batch[1]), len(batch[2])
                        feed = self.create_feed_dict(batch[0], batch[3], batch[2])#, embedding_value)
                        train, loss_ = sess.run([train_op, loss], feed_dict = feed)

                        batch_losses.append(loss_)
                        print loss_
                    print batch_losses
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
    embedding_wrapper = EmbeddingWrapper()
    embedding_wrapper.build_vocab()
    embedding_wrapper.process_glove()

    model = SequencePredictor(5, 50, embedding_wrapper)
    losses = model.fit_model()
    print losses

if __name__ == "__main__":
    main()


