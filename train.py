from embedding_wrapper import EmbeddingWrapper
from read_in_datafile import file_generator
import os
from batch_generator import batch_generator
import numpy as np

import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope as vs

# import seq2seq_model


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
        self.hidden_size = 32
        self.buckets = [(10, 10)]  
        self.num_layers = 1
        self.max_gradient_norm = 5.0
        self.learning_rate_decay_factor = 0.99
        self.batch_size = 1

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

    def pointer_decoder(self, decoder_inputs, initial_state, attention_states, cell,
                        feed_prev=True, dtype=dtypes.float32, scope=None):
        """RNN decoder with pointer net for the sequence-to-sequence model.
        Args:
          decoder_inputs: a list of 2D Tensors [batch_size x cell.input_size].
          initial_state: 2D Tensor [batch_size x cell.state_size].
          attention_states: 3D Tensor [batch_size x attn_length x attn_size].
          cell: rnn_cell.RNNCell defining the cell function and size.
          dtype: The dtype to use for the RNN initial state (default: tf.float32).
          scope: VariableScope for the created subgraph; default: "pointer_decoder".
        Returns:
          outputs: A list of the same length as decoder_inputs of 2D Tensors of shape
            [batch_size x output_size]. These represent the generated outputs.
            Output i is computed from input i (which is either i-th decoder_inputs.
            First, we run the cell
            on a combination of the input and previous attention masks:
              cell_output, new_state = cell(linear(input, prev_attn), prev_state).
            Then, we calculate new attention masks:
              new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
            and then we calculate the output:
              output = linear(cell_output, new_attn).
          states: The state of each decoder cell in each time-step. This is a list
            with length len(decoder_inputs) -- one item for each time-step.
            Each item is a 2D Tensor of shape [batch_size x cell.state_size].
        """
        # if not decoder_inputs:
        #     raise ValueError("Must provide at least 1 input to attention decoder.")
        # if not attention_states.get_shape()[1:2].is_fully_defined():
        #     raise ValueError("Shape[1] and [2] of attention_states must be known: %s"
        #                      % attention_states.get_shape())

        with vs.variable_scope(scope or "point_decoder"):
            batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
            input_size = decoder_inputs[0].get_shape()[1].value
            attn_length = attention_states.get_shape()[1].value
            attn_size = attention_states.get_shape()[2].value

            # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
            hidden = array_ops.reshape(
                attention_states, [-1, attn_length, 1, attn_size])

            attention_vec_size = attn_size  # Size of query vectors for attention.
            k = vs.get_variable("AttnW", [1, 1, attn_size, attention_vec_size])
            hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            v = vs.get_variable("AttnV", [attention_vec_size])

            states = [initial_state]

            def attention(query):
                """Point on hidden using hidden_features and query."""
                with vs.variable_scope("Attention"):
                    y = rnn_cell._linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(
                        v * math_ops.tanh(hidden_features + y), [2, 3])
                    return s

            outputs = []
            prev = None
            batch_attn_size = array_ops.pack([batch_size, attn_size])
            attns = array_ops.zeros(batch_attn_size, dtype=dtype)

            attns.set_shape([None, attn_size])
            inps = []
            for i in xrange(len(decoder_inputs)):
                if i > 0:
                    vs.get_variable_scope().reuse_variables()
                inp = decoder_inputs[i]

                if feed_prev and i > 0:
                    inp = tf.pack(decoder_inputs)
                    inp = tf.transpose(inp, perm=[1, 0, 2])
                    inp = tf.reshape(inp, [-1, attn_length, input_size])
                    inp = tf.reduce_sum(inp * tf.reshape(tf.nn.softmax(output), [-1, attn_length, 1]), 1)
                    inp = tf.stop_gradient(inp)
                    inps.append(inp)

                # Use the same inputs in inference, order internaly

                # Merge input and previous attentions into one vector of the right size.
                x = rnn_cell._linear([inp, attns], cell.output_size, True)
                # Run the RNN.
                cell_output, new_state = cell(x, states[-1])
                states.append(new_state)
                # Run the attention mechanism.
                output = attention(new_state)

                outputs.append(output)

        return outputs, states, inps


    # def create_feed_dict(self, inputs_batch, embedding_value, labels_batch=None):
    #     """Creates the feed_dict for the model.
    #     NOTE: You do not have to do anything here.
    #     """
    #     feed_dict = {
    #         self.bill_input: inputs_batch,
    #         # self.embedding_init : embedding_value
    #         }
    #     if labels_batch is not None:
    #         feed_dict[self.summary_input] = labels_batch
    #     return feed_dict

    def create_feed_dict(self, encoder_input_data, decoder_input_data, decoder_target_data):
        feed_dict = {}
        for placeholder, data in zip(self.encoder_inputs, encoder_input_data):
            feed_dict[placeholder] = data

        for placeholder, data in zip(self.decoder_inputs, decoder_input_data):
            feed_dict[placeholder] = data

        for placeholder, data in zip(self.decoder_targets, decoder_target_data):
            feed_dict[placeholder] = data

        for placeholder in self.target_weights:
            feed_dict[placeholder] = np.ones([self.batch_size, 1])

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

    def fitter_model(self):
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.decoder_targets = []
        self.target_weights = []
        cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)


        for i in xrange(self.bill_length):
            self.encoder_inputs.append(tf.placeholder(tf.float32, shape = (self.batch_size, self.glove_dim), name = 'bill_input%d'%i))


        for i in xrange(self.summ_length + 1):            
            self.decoder_inputs.append(tf.placeholder(tf.float32, shape = (self.batch_size, self.glove_dim), name = 'summ_input%d'%i))
            self.decoder_targets.append(tf.placeholder(tf.float32, shape = (self.batch_size, self.glove_dim), name = 'dec_target%d'%i))

        encoder_outputs, final_state = tf.nn.rnn(cell, self.encoder_inputs, dtype = tf.float32)
        # Need a dummy output to point on it. End of decoding.
        encoder_outputs = [tf.zeros([self.batch_size, self.hidden_size])] + encoder_outputs

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [tf.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs]
        attention_states = tf.concat(1, top_states)
        outputs = 0
        with tf.variable_scope("decoder"):
            outputs, states, _ = self.pointer_decoder(
                self.decoder_inputs, final_state, attention_states, cell)

        # with tf.variable_scope("decoder", reuse=True):
        #     predictions, _, inps = pointer_decoder(
        #         self.decoder_inputs, final_state, attention_states, cell, feed_prev=True)
            



        loss = 0.0
        print outputs
        print self.decoder_targets
        for output, target, weight in zip(outputs, self.decoder_targets, self.target_weights):
            loss += tf.nn.softmax_cross_entropy_with_logits(output, target) * weight

        loss = tf.reduce_mean(loss)
        print loss

        test_loss = 0.0
        # for output, target, weight in zip(predictions, self.decoder_targets, self.target_weights):
        #     test_loss += tf.nn.softmax_cross_entropy_with_logits(output, target) * weight

        # test_loss = tf.reduce_mean(test_loss)

        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss)
        
        train_loss_value = 0.0
        test_loss_value = 0.0
        
        correct_order = 0
        all_order = 0

        with tf.Session() as sess:
            init = tf.initialize_all_variables()
            sess.run(init)
            for i in xrange(self.num_epochs):
                batch_losses = []
                for batch in batch_generator(self.embedding_wrapper, bill_data_path, summary_data_path, BATCH_SIZE, self.bill_length, self.summ_length):
                    for padded_bill, padded_summary in batch:
                        self.create_feed_dict(padded_bill, padded_summary, padded_summary)
                        d_x, l = sess.run([loss, train_op], feed_dict=feed_dict)
                        train_loss_value = 0.9 * train_loss_value + 0.1 * d_x
                        print("Train: ", train_loss_value)







def main():
    bills_datapath = os.getcwd() + all_bill_directory
    gold_summaries_datapath = os.getcwd() + all_summary_directory

    embedding_wrapper = EmbeddingWrapper(bill_data_path)
    embedding_wrapper.build_vocab()
    embedding_wrapper.process_glove()
    # dict_obj = pickle.load(open('vocab.dat', 'r'))

    model = SequencePredictor(1, 50, embedding_wrapper)
    losses = model.fitter_model()
    print losses

if __name__ == "__main__":
    main()



