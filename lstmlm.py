import getpass
import sys
import time
import re

import numpy as np
from copy import deepcopy

from utils import calculate_perplexity, get_ptb_dataset, Vocab, get_datafile
from utils import ptb_iterator, sample, findMostPossible

import tensorflow as tf
#from tensorflow.python.ops.seq2seq import sequence_loss
# from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import sequence_loss
# from tensorflow.contrib.seq2seq.ops.seq2seq import sequence_loss
from model import LanguageModel

# Let's set the parameters of our model
# http://arxiv.org/pdf/1409.2329v4.pdf shows parameters that would achieve near
# SotA numbers

# from preprocess import PreProcess

import nltk


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    batch_size = 64   # 64
    embed_size = 200  # 50  # 50
    hidden_size = 200  # 100
    num_steps = 35
    max_epochs = 4  # 5
    early_stopping = 2
    dropout = 0.9
    # lr = 0.001
    max_grad_norm = 5
    is_training = True
    num_layers = 2
    learning_rate = 1.0
    lr_decay = 0.5

    init_scale = 0.1

    max_max_epochs = 12


class LSTMLM_Model(LanguageModel):

    def load_own_data(self, filename, filename2, filename3, debug=False, encoding='utf-8'):
        """Loads starter word-vectors and train/dev/test data."""
        self.vocab = Vocab()
        self.vocab.construct(get_datafile(filename))
        # self.vocab.construct(get_ptb_dataset('train'))
        self.encoded_train = np.array(
            [self.vocab.encode(word) for word in get_datafile(
                filename, encoding=encoding)],
            dtype=np.int32)
        self.encoded_valid = np.array(
            [self.vocab.encode(word) for word in get_datafile(
                filename2, encoding=encoding)],
            dtype=np.int32)
        self.encoded_test = np.array(
            [self.vocab.encode(word) for word in get_datafile(
                filename3, encoding=encoding)],
            dtype=np.int32)
        if debug:
            num_debug = 1024
            self.encoded_train = self.encoded_train[:num_debug]
            self.encoded_valid = self.encoded_valid[:num_debug]
            self.encoded_test = self.encoded_test[:num_debug]

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(
            tf.int32, [None, self.config.num_steps], name='Input')
        self.labels_placeholder = tf.placeholder(
            tf.int32, [None, self.config.num_steps], name='Target')
        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")

    def add_projection(self, rnn_outputs):
        with tf.variable_scope('Projection'):
            U = tf.get_variable(
                'Matrix', [self.config.hidden_size, len(self.vocab)])
            proj_b = tf.get_variable('Bias', [len(self.vocab)])
            outputs = [tf.matmul(o, U) + proj_b for o in rnn_outputs]
        # END YOUR CODE
        return outputs

    def add_embedding(self):
        with tf.device('/cpu:0'):
            embedding = tf.get_variable(
                'Embedding',
                [len(self.vocab), self.config.embed_size], trainable=True)
            inputs = tf.nn.embedding_lookup(embedding, self.input_placeholder)
            # inputs = [
            #     tf.squeeze(x, [1]) for x in tf.split(inputs, self.config.num_steps, 1)]
            return inputs

    def add_projection(self, lstm_output):
        with tf.variable_scope('Projection'):
            size = self.config.hidden_size
            vocab_size = self.vocab.__len__()
            softmax_w = tf.get_variable(
                "softmax_w", [size, vocab_size], dtype=data_type())
            softmax_b = tf.get_variable(
                "softmax_b", [vocab_size], dtype=data_type())
            logits = tf.nn.xw_plus_b(lstm_output, softmax_w, softmax_b)
            # Reshape logits to be a 3-D tensor for sequence loss
            logits = tf.reshape(
                logits, [self.config.batch_size, self.config.num_steps, vocab_size])
        return logits

    def add_loss_op(self, output):
        # Use the contrib sequence loss and average over the batches
        # all_ones = [tf.ones([self.config.batch_size * self.config.num_steps])]
        # cross_entropy = sequence_loss(
        #     output, [tf.reshape(self.labels_placeholder, [-1])], all_ones, len(self.vocab))
        # [tf.reshape(self.labels_placeholder, [-1])],
        # cost = tf.reduce_sum(cross_entropy)
        loss_1 = tf.contrib.seq2seq.sequence_loss(
            output,
            self.labels_placeholder,
            tf.ones([self.config.batch_size, self.config.num_steps],
                    dtype=data_type()),
            average_across_timesteps=False,
            average_across_batch=True)
        self.cost = tf.reduce_sum(loss_1)
        tf.add_to_collection('total_loss', self.cost)
        loss = tf.add_n(tf.get_collection('total_loss'))
        # END YOUR CODE
        return loss

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def add_training_op(self):
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          self.config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.train.get_or_create_global_step())
        self._lr_update = tf.assign(self._lr, self._new_lr)
        # optimizer = tf.train.AdamOptimizer(self.config.lr)
        # train_op = optimizer.minimize(self.calculate_loss)
        return train_op

    def _get_lstm_cell(self, is_training):
        return tf.contrib.rnn.BasicLSTMCell(
            self.config.hidden_size, forget_bias=0.0, state_is_tuple=True, reuse=not is_training)

    def add_model(self, inputs, is_training):
        '''
        Create the LSTM model
        '''
        print(inputs.shape)
        with tf.variable_scope('InputDropout'):
            if is_training and self.config.dropout < 1:
                inputs = tf.nn.dropout(inputs, self.config.dropout)
        with tf.variable_scope('LSTMMODEL') as scope:
            def make_cell():
                cell = self._get_lstm_cell(is_training)
                if is_training and self.config.dropout < 1:
                    cell = tf.contrib.rnn.DropoutWrapper(
                        cell, output_keep_prob=self.config.dropout)
                return cell

            cell = tf.contrib.rnn.MultiRNNCell(
                [make_cell() for _ in range(self.config.num_layers)], state_is_tuple=True)
            self.initial_state = cell.zero_state(
                self.config.batch_size, data_type())
            state = self.initial_state
            # inputs = tf.unstack(inputs, num=self.config.num_steps, axis=1)
            # outputs, state = tf.nn.static_rnn(
            #     cell, inputs, initial_state=self.initial_state)
            outputs = []
            with tf.variable_scope("RNNV"):
                for time_step in range(self.config.num_steps):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = cell(inputs[:, time_step, :], state)
                    outputs.append(cell_output)
            output = tf.reshape(tf.concat(outputs, 1),
                                [-1, self.config.hidden_size])
            # return output, state
            # outputs, states = tf.nn.dynamic_rnn(
            #     cell, inputs, dtype=tf.float32)
            self.final_state = state
            return output

    def run_epoch(self, session, data, train_op=None, verbose=10):
        config = self.config
        dp = config.dropout
        if not train_op:
            train_op = tf.no_op()
            dp = 1
        total_steps = sum(1 for x in ptb_iterator(
            data, config.batch_size, config.num_steps))
        # total_loss = []
        # state = self.initial_state.eval()
        costs = 0.0
        iters = 0
        for step, (x, y) in enumerate(
                ptb_iterator(data, config.batch_size, config.num_steps)):
            # We need to pass in the initial state and retrieve the final state to give
            # the RNN proper history #self.initial_state: state,
            feed = {self.input_placeholder: x,
                    self.labels_placeholder: y,
                    self.dropout_placeholder: dp}
            loss, state, cost, _ = session.run(
                [self.calculate_loss, self.final_state, self.cost, train_op], feed_dict=feed)
            # total_loss.append(loss)
            costs += cost
            iters += self.config.num_steps
            if verbose and step % verbose == 0:
                sys.stdout.write('\r{} / {} : pp = {}'.format(
                    step, total_steps, np.exp(costs / iters)))
                # sys.stdout.write('\r{} / {} : pp = {}'.format(
                #     step, total_steps, np.exp(np.mean(total_loss))))
                sys.stdout.flush()
        if verbose:
            sys.stdout.write('\r')
        # return np.exp(np.mean(total_loss))
        return np.exp(costs / iters)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def __init__(self, config):
        self.config = config
        dirname = "./data/"
        self.load_own_data(
            filename=dirname+"train_data", filename2=dirname+"dev_data", filename3=dirname+"test_data", debug=False, encoding='Latin-1')
        self.add_placeholders()

        # self._lr = tf.Variable(0.0, trainable=False)
        # self._lr_update = tf.assign(self._lr, self._new_lr)

        self.inputs = self.add_embedding()
        self.lstm_outputs = self.add_model(
            self.inputs, self.config.is_training)
        self.outputs = self.add_projection(self.lstm_outputs)

        vocab_size = self.vocab.__len__()
        logits2 = tf.reshape(
            self.outputs, [self.config.batch_size * self.config.num_steps, vocab_size])
        local_pred = tf.nn.softmax(tf.cast(logits2, tf.float64))
        local_pred2 = tf.reshape(
            local_pred, [self.config.batch_size, self.config.num_steps, vocab_size])
        self.predictions = tf.transpose(local_pred2, [1, 0, 2])

        self.calculate_loss = self.add_loss_op(self.outputs)
        self.train_step = self.add_training_op()


def data_type():
    return tf.float32


def generate_sentence(session, model, config, *args, **kwargs):
    """Convenice to generate a sentence from the model."""
    return generate_text(session, model, config, *args, stop_tokens=['<eos>'], **kwargs)


def generate_sentence2(session, model, config, options, *args, **kwargs):
    """Convenice to generate a sentence from the model."""
    return generate_next_word(session, model, config, options, *args, stop_tokens=['<eos>'], **kwargs)


def generate_next_word(session, model, config, options, starting_text='<eos>',
                       stop_length=1, stop_tokens=None, temp=1.0):
    """Generate text from the model.

    Hint: Create a feed-dictionary and use sess.run() to execute the model. Note
          that you will need to use model.initial_state as a key to feed_dict
    Hint: Fetch model.final_state and model.predictions[-1]. (You set
          model.final_state in add_model() and model.predictions is set in
          __init__)
    Hint: Store the outputs of running the model in local variables state and
          y_pred (used in the pre-implemented parts of this function.)

    Args:
      session: tf.Session() object
      model: Object of type RNNLM_Model
      config: A Config() object
      starting_text: Initial text passed to model.
    Returns:
      output: List of word idxs
    """
    # state = model.initial_state.eval()
    # Imagine tokens as a batch size of one, length of len(tokens[0])
    tokens = [model.vocab.encode(word) for word in starting_text.split()]
    options = [model.vocab.encode(word) for word in options]
    next_word_list = []
    for i in range(stop_length):
        # YOUR CODE HERE
        feed = {model.input_placeholder: [tokens[-1:]],
                model.dropout_placeholder: 1}  # model.initial_state: state,
        state, y_pred = session.run(
            [model.final_state, model.predictions[-1]], feed_dict=feed)
        # END YOUR CODE
        # next_word_idx = sample(y_pred[0], temperature=temp)
        next_word_idx = findMostPossible(y_pred[0], options, temperature=temp)
        # print("options")
        # print(options)
        # print(next_word_idx)
        tokens.append(next_word_idx)
        next_word_list.append(next_word_idx)
        if stop_tokens and model.vocab.decode(tokens[-1]) in stop_tokens:
            break
    # output = [model.vocab.decode(word_idx) for word_idx in tokens]
    # return output
    return model.vocab.decode(next_word_idx)


def test_RNNLM():
    config = Config()
    gen_config = deepcopy(config)
    gen_config.batch_size = gen_config.num_steps = 1
    gen_config.is_training = False

    initializer = tf.random_uniform_initializer(
        -config.init_scale, config.init_scale)
    # We create the training model and generative model
    with tf.variable_scope('RNNLM', reuse=tf.AUTO_REUSE, initializer=initializer) as scope:
        model = LSTMLM_Model(config)
        # This instructs gen_model to reuse the same variables as the model above
        # scope.reuse_variables()
        gen_model = LSTMLM_Model(gen_config)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session() as session:
        best_val_pp = float('inf')
        best_val_epoch = 0

        session.run(init)
        for epoch in range(config.max_max_epochs):
            print('Epoch {}'.format(epoch))
            start = time.time()
            ###
            lr_decay = config.lr_decay ** max(epoch +
                                              1 - config.max_epochs, 0.0)
            model.assign_lr(session, config.learning_rate * lr_decay)
            print("Epoch: %d Learning rate: %.3f" %
                  (epoch, session.run(model._lr)))

            model.config.is_training = True
            train_pp = model.run_epoch(
                session, model.encoded_train,
                train_op=model.train_step)
            print('Training perplexity: {}'.format(train_pp))

            model.config.is_training = False
            valid_pp = model.run_epoch(session, model.encoded_valid)
            print('Validation perplexity: {}'.format(valid_pp))
            if valid_pp < best_val_pp:
                best_val_pp = valid_pp
                best_val_epoch = epoch
                saver.save(session, './ptb_lstmlm.weights')
            if epoch - best_val_epoch > config.early_stopping:  # stop early_stopping
                break
            print('Total time: {}'.format(time.time() - start))

        saver.restore(session, 'ptb_lstmlm.weights')
        model.config.is_training = False
        test_pp = model.run_epoch(session, model.encoded_test)
        print('=-=' * 5)
        print('Test perplexity: {}'.format(test_pp))
        print('=-=' * 5)

        # starting_text = 'in palo alto'
        # starting_text = 'We are now'
        # while starting_text:
        #     print(' '.join(generate_sentence(
        #         session, gen_model, gen_config, starting_text=starting_text, temp=1.0)))
        #     starting_text = input('> ')
        # with open('/home/nlp/project2_homework/data/development_set.txt', 'rt', encoding='Latin-1') as df:
        #     data = df.read()
        #     dev_list = re.findall(
        #         r"(\d+)\)(.+$)\s+(a)\)\s+(\w+)$\s+(b)\)\s+(\w+)$\s+(c)\)\s+(\w+)$\s+(d)\)\s+(\w+)$\s+(e)\)\s+(\w+)$", data, re.M)
        # devel_cloze_right_ans = []
        # with open('/home/nlp/project2_homework/data/development_set_answers.txt', 'rt', encoding='Latin-1') as df:
        #     data = df.read()
        #     devel_cloze_right_ans = re.findall(
        #         r"(\d+)\)\s+\[(\w)\]\s+(\w+)$", data, re.M)
        with open('./data/development_set.txt', 'rt', encoding='Latin-1') as df:
            data = df.read()
            dev_list = re.findall(
                r"(\d+)\)(.+$)\s+(a)\)\s+([0-9a-zA-Z_',-]+)$\s+(b)\)\s+([0-9a-zA-Z_,'-]+)$\s+(c)\)\s+([0-9a-zA-Z_,'-]+)$\s+(d)\)\s+([0-9a-zA-Z_,'-]+)$\s+(e)\)\s+([0-9a-zA-Z_,'-]+)$", data, re.M)
        devel_cloze_right_ans = []
        with open('./data/development_set_answers.txt', 'rt', encoding='Latin-1') as df:
            data = df.read()
            devel_cloze_right_ans = re.findall(
                r"(\d+)\)\s+\[([a-zA-Z_'-])\]\s+([a-zA-Z_'-]+)$", data, re.M)
        y = np.array(devel_cloze_right_ans)
        next_word = []
        y_hat = []
        print(len(dev_list))
        print(len(devel_cloze_right_ans))
        # sentence_unk_word = 0
        # options_unk_word = 0

        for question in dev_list:
            ques = question[1]
            ques_front = ques.split("_____")
            for i in range(len(ques_front)):
                ques_front[i].strip()
            options = []
            for i in range(3, 12, 2):
                options.append(question[i])
            nword = generate_sentence2(
                session, gen_model, gen_config, options, starting_text=' '.join(nltk.word_tokenize(ques_front[0])), temp=1.0)
            # print("nword type")
            # print(type(nword))
            y_hat.append(nword)

        #     for s in nltk.word_tokenize(ques_front[0]):
        #         # for s in ques_front[0].split():
        #         if(model.vocab.encode(s) == 0):
        #             print("sentence " + s)
        #             sentence_unk_word = sentence_unk_word + 1

        #     for s in options:
        #         if(model.vocab.encode(s) == 0):
        #             print("option " + s)
        #             options_unk_word = options_unk_word + 1

        # print("unk = "+str(sentence_unk_word+options_unk_word))

        y_hat = np.array(y_hat)  # .reshape(-1, 1)
        print(y_hat.shape)
        print(y[:, 2].shape)
        accuracy = np.mean(y[:, 2] == y_hat)
        print('accuracy = %f' % (accuracy))
        # for i in range(len(y_hat)):
        print(y_hat)
        print(y[:, 2])


def check_unk_numbers():
    config = Config()
    # with tf.variable_scope('LSTMLM', reuse=tf.AUTO_REUSE) as scope:
    model = LSTMLM_Model(config)
    # init = tf.initialize_all_variables()
    # with tf.Session() as session:
    with open('/home/nlp/project2_homework/data/development_set.txt', 'rt', encoding='Latin-1') as df:
        data = df.read()
        dev_list = re.findall(
            r"(\d+)\)(.+$)\s+(a)\)\s+([0-9a-zA-Z_',-]+)$\s+(b)\)\s+([0-9a-zA-Z_,'-]+)$\s+(c)\)\s+([0-9a-zA-Z_,'-]+)$\s+(d)\)\s+([0-9a-zA-Z_,'-]+)$\s+(e)\)\s+([0-9a-zA-Z_,'-]+)$", data, re.M)
        # devel_cloze_right_ans = []
        # with open('/home/nlp/project2_homework/data/development_set_answers.txt', 'rt', encoding='Latin-1') as df:
        #     data = df.read()
        #     devel_cloze_right_ans = re.findall(
        #         r"(\d+)\)\s+\[([a-zA-Z_'-])\]\s+([a-zA-Z_'-]+)$", data, re.M)
        # y = np.array(devel_cloze_right_ans)
    sentence_unk_word = 0
    options_unk_word = 0
    for question in dev_list:
        ques = question[1]
        ques_front = ques.split("_____")
        # for i in range(len(ques_front)):
        #     ques_front[i].strip()
        options = []
        for i in range(3, 12, 2):
            options.append(question[i])
        # nword = generate_sentence2(
        #     session, gen_model, gen_config, options, starting_text=' '.join(nltk.word_tokenize(ques_front[0])), temp=1.0)
        # print("nword type")
        # print(type(nword))

        for s in nltk.word_tokenize(ques_front[0]):
            # for s in ques_front[0].split():
            if(model.vocab.encode(s) == 0):
                print("sentence " + s)
                sentence_unk_word = sentence_unk_word
        for s in options:
            if(model.vocab.encode(s) == 0):
                print("option " + s)
                options_unk_word = options_unk_word + 1

    print("unk = "+str(sentence_unk_word+options_unk_word))


if __name__ == "__main__":
    test_RNNLM()
    # check_unk_numbers()
