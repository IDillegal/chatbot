from __future__ import absolute_import, division, print_function

import os
import time

import numpy as np
import tensorflow as tf

from model import *

# Enable eager execution
tf.enable_eager_execution()

batch_size = 64
grad_clip = 5
learning_rate = 0.001  # When the loss stops diminishing, then we reduce the learning_rate by 2 times
epochs = 20
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# Load dataset to input and target tensors. Creating or reading the vocabulary
input_tensor, target_tensor, vocab = load_dataset(path_to_dataset, path_to_vocab, max_length)

vocab_size = len(vocab.word2idx)
buffer_size = len(input_tensor)
n_batch = buffer_size // batch_size
dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(buffer_size)
dataset = dataset.batch(batch_size, drop_remainder=True)

encoder = Encoder(vocab_size, embedding_dim, units, num_layers, dropout_keep_prob)
decoder = Decoder(vocab_size, embedding_dim, units, num_layers, dropout_keep_prob)

# Define the optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
# Checkpoints (Object-based saving)
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


# Define the loss function
def loss_func(real, predict):
    mask = 1 - np.equal(real, 0)  # masking the loss calculated for padding
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=predict) * mask
    return tf.reduce_mean(loss_)


# Restoring the latest checkpoint in checkpoint_dir
if os.path.exists(os.path.join(checkpoint_dir, "checkpoint")):
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Training
for epoch in range(epochs):
    start = time.time()
    hidden_init = encoder.initialize_hidden_state(batch_size)
    total_loss = 0
    for (batch, (input, target)) in enumerate(dataset):
        loss = 0
        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(input, hidden_init)
            dec_hidden = enc_hidden
            # Add time axis, shape == (batch_size, 1)
            dec_input = tf.expand_dims([vocab.word2idx['<start>']] * batch_size, 1)
            # Teacher forcing - feeding the target as the next input
            for t in range(1, target.shape[1]):
                # Passing enc_output to the decoder
                predict, dec_hidden = decoder(dec_input, dec_hidden, enc_output)
                loss += loss_func(target[:, t], predict)
                # Using teacher forcing. Add time axis, shape == (batch_size, 1)
                dec_input = tf.expand_dims(target[:, t], 1)
        batch_loss = loss / int(target.shape[1])
        total_loss += batch_loss

        # Calculate the gradients and apply it to the optimizer and backpropagate
        vars = encoder.variables + decoder.variables
        # Gradient clipping to avoid "exploding gradients"
        grads, _ = tf.clip_by_global_norm(tape.gradient(loss, vars), grad_clip)
        optimizer.apply_gradients(zip(grads, vars))

        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
        encoder.save_weights(encoder_checkpoint_prefix)
        decoder.save_weights(decoder_checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / n_batch))
    print('Time taken for 1 epoch {:.2f} sec\n'.format(time.time() - start))
