from embedding_wrapper import EmbeddingWrapper
from read_in_datafile import file_generator
import os
from batch_generator import batch_generator
import numpy as np
from encoder_decoder_cells import DecoderCell

import tensorflow as tf

bill_data_file = "bills_data_100_test.txt"
summary_data_file = "extracted_data_full.txt"
indices_data_file = "indices_data_100_test.txt"
sequence_data_file = "sequence_lengths.txt"

BATCH_SIZE = 1

class SequencePredictor():
    def __init__(self, num_epochs, glove_dim, embedding_wrapper):
        self.glove_dim = glove_dim
        self.num_epochs = num_epochs
        self.bill_length = 100
        self.lr = 0.05
        self.inputs_placeholder = None
        self.summary_input = None
        self.mask_placeholder = None
        self.hidden_size = 10

        self.labels_placeholder = None
        self.embedding_wrapper = embedding_wrapper
        self.vocab_size = embedding_wrapper.num_tokens
        self.embedding_init = None

    def consolidate_predictions(self, examples_raw, examples, preds):
        """Batch the predictions into groups of sentence length.
        """
        assert len(examples_raw) == len(examples)
        assert len(examples_raw) == len(preds)

        ret = []
        for i, (sentence, labels) in enumerate(examples_raw):
            _, _, mask = examples[i]
            labels_ = [l for l, m in zip(preds[i], mask) if m] # only select elements of mask.
            assert len(labels_) == len(labels)
            ret.append([sentence, labels, labels_])
        return ret

    def evaluate(self, sess, examples, examples_raw):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            examples: A list of vectorized input/output pairs.
            examples: A list of the original input/output sequence pairs.
        Returns:
            The F1 score for predicting tokens as named entities.
        """
        token_cm = ConfusionMatrix(labels=LBLS)

        correct_preds, total_correct, total_preds = 0., 0., 0.
        for _, labels, labels_  in self.output(sess, examples_raw, examples):
            for l, l_ in zip(labels, labels_):
                token_cm.update(l, l_)
            gold = set(get_chunks(labels))
            pred = set(get_chunks(labels_))
            correct_preds += len(gold.intersection(pred))
            total_preds += len(pred)
            total_correct += len(gold)

        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        return token_cm, (p, r, f1)

    def output(self, sess, inputs_raw, inputs=None):
        """
        Reports the output of the model on examples (uses helper to featurize each example).
        """
        if inputs is None:
            inputs = self.preprocess_sequence_data(self.helper.vectorize(inputs_raw))

        preds = []
        prog = Progbar(target=1 + int(len(inputs) / self.config.batch_size))
        for i, batch in enumerate(minibatches(inputs, self.config.batch_size, shuffle=False)):
            # Ignore predict
            batch = batch[:1] + batch[2:]
            preds_ = self.predict_on_batch(sess, *batch)
            preds += list(preds_)
            prog.update(i + 1, [])
        return self.consolidate_predictions(inputs_raw, inputs, preds)

    def create_feed_dict(self, inputs_batch, masks_batch, sequences, labels_batch = None):
        feed_dict = {
            self.inputs_placeholder : inputs_batch,
            self.mask_placeholder : masks_batch,
            self.sequences_placeholder : sequences
            }
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def fit_model(self):
        with tf.Graph().as_default():
            # add placeholders
            self.inputs_placeholder = tf.placeholder(tf.int32, shape=(None, self.bill_length))
            self.mask_placeholder = tf.placeholder(tf.bool, shape=(None, self.bill_length))
            self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.bill_length))
            self.sequences_placeholder = tf.placeholder(tf.int32, shape=(BATCH_SIZE))
            # add embeddings
            data = np.load('trimmed_glove.6B.50d.npz')
            embeddings = tf.Variable(data['glove'])
            bill_embeddings = tf.nn.embedding_lookup(embeddings, self.inputs_placeholder)
            bill_embeddings = tf.reshape(bill_embeddings, (-1, self.bill_length, self.glove_dim))        
            #encoder
            with tf.variable_scope("encoder"):
                enc_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
                outputs, state = tf.nn.dynamic_rnn(enc_cell,bill_embeddings, dtype = tf.float64)
            #decoder
            preds = []
            U = tf.get_variable('U', (self.hidden_size * self.bill_length, self.bill_length), initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
            b2 = tf.get_variable('b2', (self.bill_length), \
            initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float64)
            with tf.variable_scope("decoder"):
                dec_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size)
                preds, state = tf.nn.dynamic_rnn(dec_cell,bill_embeddings, sequence_length = self.sequences_placeholder, dtype = tf.float64, initial_state = state)
                print preds
            #print preds[0]
            # print preds
            # preds = tf.concat(preds, 1)
            #preds = tf.pack(preds)
            #preds = tf.transpose(preds, [1,0,2])
            preds = tf.reshape(preds, (-1, self.hidden_size * self.bill_length))
            print preds
            preds = tf.matmul(preds, U) + b2
            # print pred
            # print tf.nn.softmax(preds)
            # print
            # preds = tf.nn.softmax(preds)
            #self.labels_placeholder = tf.cast(self.labels_placeholder, tf.int32)
            print self.labels_placeholder
            loss = tf.nn.softmax_cross_entropy_with_logits(preds, self.labels_placeholder)
            #masked_loss = tf.boolean_mask(loss, self.mask_placeholder)
            loss = tf.reduce_mean(loss)

            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            train_op = optimizer.minimize(loss)

            epoch_losses = []
            best_score = 0
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())
                for i in xrange(self.num_epochs):
                    print 'EPOCH %s of %d' % (i + 1, self.num_epochs)
                    batch_losses = []
                    batch_counter = 0
                    for inputs,labels, masks, sequences in batch_generator(self.embedding_wrapper, bill_data_file, indices_data_file, sequence_data_file, BATCH_SIZE, self.bill_length):
                        print 'BATCH %s' % (batch_counter)
                        feed = self.create_feed_dict(inputs, masks, sequences, labels_batch = labels)
                        train, loss_ = sess.run([train_op, loss], feed_dict = feed)
                        print "train: ", train
                        batch_losses.append(loss_)
                        print loss_
                        batch_counter += 1
                    print batch_losses
                    epoch_losses.append(batch_losses)

                    logger.info("Evaluating on development data")
                    token_cm, entity_scores = self.evaluate(sess, dev_set, dev_set_raw)
                    #logger.debug("Token-level confusion matrix:\n" + token_cm.as_table())
                    logger.debug("Token-level scores:\n" + token_cm.summary())
                    logger.info("Entity level P/R/F1: %.2f/%.2f/%.2f", *entity_scores)

                    f1 = entity_scores[-1]

            return epoch_losses

def main():
    embedding_wrapper = EmbeddingWrapper()
    embedding_wrapper.build_vocab()
    embedding_wrapper.process_glove()

    model = SequencePredictor(10, 50, embedding_wrapper)
    losses = model.fit_model()
    print losses

if __name__ == "__main__":
    main()


