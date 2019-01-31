from __future__ import absolute_import, division, print_function

import logging

from telegram.ext import BaseFilter, Filters, CommandHandler, MessageHandler, Updater
import tensorflow as tf

from model import *

# Enable eager execution
tf.enable_eager_execution()

# Here should be your token
TOKEN = '<TOKEN>'

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# Reading the vocabulary
vocab = WordIndex(path_to_vocab)
vocab_size = len(vocab.word2idx)

encoder = Encoder(vocab_size, embedding_dim, units, num_layers, dropout_keep_prob)
decoder = Decoder(vocab_size, embedding_dim, units, num_layers, dropout_keep_prob)

encoder.load_weights(tf.train.latest_checkpoint(encoder_checkpoint_dir))
decoder.load_weights(tf.train.latest_checkpoint(decoder_checkpoint_dir))


# Message reply function
def response(message, encoder, decoder, vocab, max_length):
    # Clean input message
    message = preprocess_message(message)

    # Vectorize the input message
    input = [vocab.word2idx.get(i, 1) for i in message.split(' ')]
    input = tf.keras.preprocessing.sequence.pad_sequences([input], maxlen=max_length, padding='post')
    input = tf.convert_to_tensor(input)

    result = ''

    hidden_init = encoder.initialize_hidden_state(1)  # batch_size = 1
    enc_out, enc_hidden = encoder(input, hidden_init)
    dec_hidden = enc_hidden
    # Add time axis, shape == (1, 1)
    dec_input = tf.expand_dims([vocab.word2idx['<start>']], 0)

    for t in range(max_length):
        predict, dec_hidden = decoder(dec_input, dec_hidden, enc_out)
        predict_id = tf.argmax(predict[0]).numpy()
        result += vocab.idx2word[predict_id] + ' '

        # Stop predicting when the model predicts the <end> token
        if vocab.idx2word[predict_id] == '<end>':
            # Clean message if model predicts the <end> token
            result = ". ".join([res.strip().capitalize() for res in
                                result.replace(" ' ", "'").replace("<end>", "").replace(".", "\n").splitlines()])
            return result

        # The predicted ID is fed back into the model. Add time axis, shape == (1, 1)
        dec_input = tf.expand_dims([predict_id], 0)
    # Clean message if max length sentence
    result = ". ".join([res.strip().capitalize() for res in
                        result.replace(" ' ", "'").replace(".", "\n").splitlines()])
    return result


# Start command
def start(bot, update):
    bot.send_message(chat_id=update.message.chat_id,
                     text='Hi, ' + update.message.from_user.first_name + '  ' + update.message.from_user.last_name + '. ' +
                          'I am HerchenovBot. You can ask about something.')


# User message handler
def handle_text(bot, update):
    logger.info("Message '%s' is received from %s %s", update.message.text, update.message.from_user.first_name,
                update.message.from_user.last_name)
    text = response(update.message.text, encoder, decoder, vocab, max_length)
    bot.send_message(chat_id=update.message.chat_id, text=text)
    logger.info("Response '%s' is sent to %s %s", text, update.message.from_user.first_name,
                update.message.from_user.last_name)


# Log Errors caused by Updates
def error(bot, update, error):
    logger.warning('Update "%s" caused error "%s"', update, error)


# Create the EventHandler and pass it your bot's token
updater = Updater(token=TOKEN)
# Get the dispatcher to register handlers
dispatcher = updater.dispatcher
# Add handlers
text_handler = MessageHandler(Filters.text, handle_text)
dispatcher.add_handler(text_handler)
dispatcher.add_handler(CommandHandler('start', start))
dispatcher.add_handler(CommandHandler('help', start))
# log all errors
dispatcher.add_error_handler(error)

# The bot is started and runs until we press Ctrl-C on the command line
updater.start_polling()
updater.idle()
