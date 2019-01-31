from __future__ import absolute_import, division, print_function

import os
import re

import tensorflow as tf

# Enable eager execution
tf.enable_eager_execution()

max_length = 40
embedding_dim = 256
units = 1024
num_layers = 2
dropout_keep_prob = 1.0  # If train 0.9, if chatbot 1.0
path_to_dataset = "./dataset.txt"
path_to_vocab = "./vocab.txt"
encoder_checkpoint_dir = './weights_encoder'
decoder_checkpoint_dir = './weights_decoder'
encoder_checkpoint_prefix = os.path.join(encoder_checkpoint_dir, "ckpt")
decoder_checkpoint_prefix = os.path.join(decoder_checkpoint_dir, "ckpt")


# PREPARE DATA

# Clean messages
def preprocess_message(m):
    # Creating a space between a word and the punctuation following it
    m = re.sub(r"([?.!,'-])", r" \1 ", m)
    m = re.sub(r'[" "]+', " ", m)
    # Replacing everything with space except (a-z, A-Z, "?", ".", "!", ",", "'", "-")
    m = re.sub(r"[^a-zA-Z0-9?.!,'-]+", " ", m)
    # All case-based characters have been lowercased
    m = m.lower().strip()
    # Add a <start> and <end> token to each message
    m = '<start> ' + m + ' <end>'
    return m


# Return pairs of messages in the format: [question, answer]
def split_dataset(path_to_dataset):
    lines = open(path_to_dataset, encoding='UTF-8').read().strip().split('\n')
    message_pairs = [[preprocess_message(m) for m in l.split('\t')] for l in lines]
    return message_pairs


# Create a word index and reverse word index
class WordIndex():
    def __init__(self, path="vocab.txt", pairs=0):
        self.pairs = pairs
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.path = path
        if os.path.exists(path):
            self.file_read()
        else:
            if pairs:
                self.create_index()
            else:
                raise Exception('No dictionary')

    def create_index(self):
        for message in self.pairs:
            self.vocab.update(message.split(' '))
        self.vocab = sorted(self.vocab)
        self.word2idx['<pad>'] = 0
        self.word2idx['<unk>'] = 1
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 2
        for word, index in self.word2idx.items():
            self.idx2word[index] = word
        self.file_write()

    def file_write(self):
        with open(self.path, "w") as f:
            for i in range(len(self.idx2word)):
                f.write(self.idx2word[i] + '\n')

    def file_read(self):
        with open(self.path, "r") as f:
            i = 0
            for word in f.read().splitlines():
                self.idx2word[i] = word
                self.word2idx[word] = i
                i += 1


# Load dataset to input and target tensors
def load_dataset(path_to_dataset, path_to_vocab, max_length):
    # Creating cleaned input, output pairs
    pairs = split_dataset(path_to_dataset)
    # Creating or reading the vocabulary
    vocab = WordIndex(path_to_vocab, (qu + " " + ans for qu, ans in pairs))
    # Vectorize the input and target messages
    # Input messages
    input_tensor = [[vocab.word2idx.get(m, 1) for m in qu.split(' ')] for qu, ans in pairs]
    # Target messages
    target_tensor = [[vocab.word2idx.get(m, 1) for m in ans.split(' ')] for qu, ans in pairs]
    # Padding the input and target tensors to the maximum length
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                 maxlen=max_length,
                                                                 padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                                  maxlen=max_length,
                                                                  padding='post')
    return input_tensor, target_tensor, vocab


# Build the multi-layer RNN cells
def gru(units, num_layers, keep_prob):
    return tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(units),
                                                                      output_keep_prob=keep_prob)
                                        for i in range(num_layers)])


# Build the encoder model
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, num_layers, keep_prob):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.enc_units, self.num_layers, self.keep_prob)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = tf.nn.dynamic_rnn(self.gru, x, initial_state=hidden)
        return output, state

    # Define the initial state
    def initialize_hidden_state(self, batch_size):
        return self.num_layers * (tf.zeros((batch_size, self.enc_units)),)


# Build the decoder model with Bahdanau attention mechanism
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, num_layers, keep_prob):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(self.dec_units, self.num_layers, self.keep_prob)
        self.fc = tf.keras.layers.Dense(vocab_size)
        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden[self.num_layers - 1], 1)
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # passing the concatenated vector to the GRU
        output, state = tf.nn.dynamic_rnn(self.gru, x, initial_state=hidden, dtype=tf.float32)
        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))
        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)
        return x, state
