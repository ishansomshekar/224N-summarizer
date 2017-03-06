import tensorflow as tf
from tensorflow.python.ops.rnn_cell import BasicLSTMCell
from tensorflow.python.ops.rnn_cell import LSTMStateTuple

k = 150
dim_embedding = 50
dim_hidden = 100



class EncoderCell(BasicLSTMCell):
    @property
    def state_size(self):
        return self._num_units

    def __call__(self, input, state, scope=None):
        """Long short-term memory (LSTM) encoder cell."""
        # Current vector and its embedding
        e = tf.matmul(tf.get_variable("we", [dim_embedding, k]), input)
        # Reset calculation
        r = tf.sigmoid(tf.matmul(tf.get_variable("wr", [dim_hidden, dim_embedding]), e)
                           + tf.matmul(tf.get_variable("ur", [dim_hidden, dim_hidden]), state))
        # Update calculation
        z = tf.sigmoid(tf.matmul(tf.get_variable("wz", [dim_hidden, dim_embedding]), e)
                           + tf.matmul(tf.get_variable("uz", [dim_hidden, dim_hidden]), state))
        # Hidden-tilde calculation
        h_tilde = tf.tanh(tf.matmul(tf.get_variable("w", [dim_hidden, dim_embedding]), e)
                              + tf.matmul(tf.get_variable("u", [dim_hidden, dim_hidden]), r * state))
        # Hidden calculation
        one = tf.ones([dim_hidden, 1])
        h = z * state + (one - z) * h_tilde
        return 0, h

class DecoderCell(BasicLSTMCell):
    def __call__(self, input, state, scope=None):
        """Long short-term memory (LSTM) decoder cell."""
        c, h_prime_previous = state
        # Current vector's embedding
        e = tf.matmul(tf.get_variable("w_prime_e", [dim_embedding, k]), input)
        # Reset calculation
        r_prime = tf.sigmoid(tf.matmul(tf.get_variable("w_prime_r", [dim_hidden, dim_embedding]), e)
                             + tf.matmul(tf.get_variable("u_prime_r", [dim_hidden, dim_hidden]), h_prime_previous)
                             + tf.matmul(tf.get_variable("Cr", [dim_hidden, dim_hidden]), c))
        # Update calculation
        z_prime = tf.sigmoid(tf.matmul(tf.get_variable("w_prime_z", [dim_hidden, dim_embedding]), e)
                             + tf.matmul(tf.get_variable("u_prime_z", [dim_hidden, dim_hidden]), h_prime_previous)
                             + tf.matmul(tf.get_variable("Cz", [dim_hidden, dim_hidden]), c))
        # Hidden-tilde calculation
        h_tilde_prime = tf.tanh(tf.matmul(tf.get_variable("w_prime", [dim_hidden, dim_embedding]), e)
                                + r_prime * (tf.matmul(tf.get_variable("u_prime", [dim_hidden, dim_hidden]), h_prime_previous)
                                + tf.matmul(tf.get_variable("C", [dim_hidden, dim_hidden]), c)))
        # Hidden calculation
        one = tf.ones([dim_hidden, 1])
        h_prime = z_prime * h_prime_previous + (one - z_prime) * h_tilde_prime
        # Maxout calculation
        s_prime = tf.matmul(tf.get_variable("oh", [2 * dim_hidden, dim_hidden]), h_prime) + tf.matmul(tf.get_variable("oy", [2 * dim_hidden, k]), input) \
                                         + tf.matmul(tf.get_variable("oc", [2 * dim_hidden, dim_hidden]), c)
        s = tf.reshape(tf.reduce_max(tf.reshape(s_prime, [dim_hidden, 2]), 1), [dim_hidden, 1])
        # Logit calculation
        g = tf.matmul(tf.get_variable("gl", [k, dim_embedding]), tf.get_variable("gr", [dim_embedding, dim_hidden]))
        logit = tf.matmul(g, s)
        return logit, LSTMStateTuple(c, h_prime)