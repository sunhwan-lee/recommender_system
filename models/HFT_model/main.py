import os
from datetime import datetime
from collections import namedtuple
import tensorflow as tf

from data import read_file, split_data, build_vocab, token_to_id
from model import HFTModel, offsetModel

FLAGS = tf.app.flags.FLAGS

# Hyperparameters
tf.app.flags.DEFINE_integer('vocab_size', 5000, 'Size of vocabulary. \
                                                  These will be read from the vocabulary file \
                                                  in order. If the vocabulary file contains \
                                                  fewer words than this number, or if this \
                                                  number is set to 0, will take all words \
                                                  in the vocabulary file.')
tf.app.flags.DEFINE_integer('emb_dim', 5, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('max_iter_steps', 50, 'max number of iterations for minimizing \
                                                   the loss')
tf.app.flags.DEFINE_integer('num_iter_steps', 50, 'number of iterations for minimizing \
                                                   the total loss')
tf.app.flags.DEFINE_float('init_stddev', 0.1, 'Standard deviation of normal distribution \
                                               to initialize variables.')
tf.app.flags.DEFINE_float('min_kappa', 0.1,  'Minimum value to initialize kappa variable')
tf.app.flags.DEFINE_float('max_kappa', 10.0, 'Maximum value to initialize kappa variable')
tf.app.flags.DEFINE_float('mu', 0.1, 'parameter that trade-offs rating error and corpus likelihood')
tf.app.flags.DEFINE_float('threshold', 1e-3,  'Threshold for the convergence of Phi and Theta variables')

# Where to find data
tf.app.flags.DEFINE_string('data_path', 'data/reviews_Automotive_5.json.gz', 'Path to source file.')
# Where to save output
tf.app.flags.DEFINE_string('log_root', 'logs/', 'Root directory for all logging.')

def main(unused_argv):

  if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
    raise Exception('Problem with flags: %s' % unused_argv)

  # choose what level of logging you want
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO) 

  # user_rating has the elements in the following order
  # user_id, item_id, rating, time, num_words, review
  user_rating, user_id_to_idx, item_id_to_idx = read_file(FLAGS.data_path)
  num_users = len(user_id_to_idx)
  num_items = len(item_id_to_idx)
  num_reviews = len(user_rating)
  print('Number of total users / items / reviews: %d / %d / %d' % 
          (num_users, num_items, num_reviews))
  users_ratings = [ur for ur in user_rating]
  train_ratings, test_ratings, valid_ratings = split_data(users_ratings)

  # build vocabulary
  id_to_word, word_to_id = build_vocab(users_ratings, FLAGS.vocab_size)
  train_item_doc = token_to_id(train_ratings, word_to_id)
  valid_item_doc = token_to_id(valid_ratings, word_to_id)

  # Try offset model
  offset_model = offsetModel(train_ratings, valid_ratings, test_ratings)
  offset_model.train()

  # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
  hparam_list = ['init_stddev', 'emb_dim', 'min_kappa', 'max_kappa', 
                 'vocab_size', 'mu', 'max_iter_steps', 'num_iter_steps',
                 'threshold']
  hps_dict = {}
  for key,val in FLAGS.flag_values_dict().items(): # for each flag
    if key in hparam_list: # if it's in the list
      hps_dict[key] = val # add it to the dict
  hps = namedtuple('HParams', hps_dict.keys())(**hps_dict)
  
  current_datetime = datetime.now()
  subfolder_timestamp = datetime.strftime(current_datetime, '%Y%m%d-%H%M%S')
  log_folder = os.path.join(FLAGS.log_root, subfolder_timestamp)

  hft_model = HFTModel(hps, train_ratings, valid_ratings, test_ratings,
                       train_item_doc, valid_item_doc,
                       num_users, num_items, num_reviews, log_folder)
  hft_model.build_graph()  
  hft_model.train()

if __name__ == '__main__':
  tf.compat.v1.app.run()