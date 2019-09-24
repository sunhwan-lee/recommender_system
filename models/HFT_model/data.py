import gzip, json
import random
from collections import defaultdict

def _parse_line(line):

  idx_start = 0
  val_cnt = 0
  for idx in range(len(line)):
    ch = line[idx]
    if ch == ' ':
      val = line[idx_start:idx]
      idx_start = idx+1
      if val_cnt == 0:
        user_id = val
      elif val_cnt == 1:
        item_id = val
      elif val_cnt == 2:
        rating = float(val)
      elif val_cnt == 3:
        time = int(val)
      elif val_cnt == 4:
        num_words = int(val)
        review = line[idx_start:].strip()
        break

      val_cnt += 1

  return [user_id, item_id, rating, time, num_words, review]

def read_file(filepath):

  print('Reading file...\n')
  # read file 
  user_rating = []
  g = gzip.open(filepath, 'r')
  for line in g:
    l = eval(line)
    user_rating.append([l['reviewerID'], l['asin'], l['overall'], 
                        l['unixReviewTime'], l['reviewText']])
      
  # build a mapping from user_id to index
  users = sorted(set([val[0] for val in user_rating]))
  user_id_to_idx = {user_id: idx for idx, user_id in enumerate(users)}

  # build a mapping from item_id to index
  items = sorted(set([val[1] for val in user_rating]))
  item_id_to_idx = {item_id: idx for idx, item_id in enumerate(items)}

  # replace user id and item id with index
  user_rating = [[user_id_to_idx[val[0]], item_id_to_idx[val[1]]] + val[2:] \
                  for val in user_rating]

  return user_rating, user_id_to_idx, item_id_to_idx

def build_vocab(ratings, vocab_size):

  print('Building vocaburaly...')
  word_cnt = defaultdict(int)
  for r in ratings:
    for w in r[-1].split():
      word_cnt[w] += 1

  word_cnt = sorted([[c,w] for w,c in word_cnt.items()], reverse=True)

  id_to_word = {idx:word[1] for idx,word in enumerate(word_cnt[:vocab_size])}
  word_to_id = {word:idx for idx,word in id_to_word.items()}

  return id_to_word, word_to_id

def token_to_id(ratings, word_to_id):

  # build documents for item
  ratings = sorted(ratings, key=lambda x:x[1])
  one_item_doc = []
  item_doc = []
  prev_item_id = None
  for ur in ratings:
    curr_item_id = ur[1]
    one_review = [word_to_id[w] for w in ur[-1].split() if w in word_to_id]
    if prev_item_id is not None and curr_item_id != prev_item_id:
      item_doc.append(one_item_doc)
      one_item_doc = one_review
    else:
      one_item_doc += one_review
    prev_item_id = curr_item_id
  item_doc.append(one_item_doc)

  return item_doc


# Utility to split the data into training, test, and valid sets.
def split_data(data, test_fraction=0.1, valid_fraction=0.1):
  """Splits data into training, test, and valid sets.
  Args:
    data: a list of examples. [user_id, item_id, rating, time, num_words, review]
    test_fraction: fraction of data to use in the test set.
    valid_fraction: fraction of data to use in the valid set.
  Returns:
    train: list of examples for training
    valid: list of examples for validation
    test: list of examples for testing
  """

  random.seed(0)
  # select test dataset from users having multiple ratings
  num_data = len(data)
  test_size = int(num_data*test_fraction)
  test_idx = random.sample(list(range(num_data)), test_size)
  valid_size = int(num_data*valid_fraction)
  valid_idx = random.sample([i for i in range(num_data) if i not in test_idx], valid_size)
  test = [d for i, d in enumerate(data) if i in test_idx]
  valid = [d for i, d in enumerate(data) if i in valid_idx]
  train = [d for i, d in enumerate(data) if i not in test_idx + valid_idx]

  users_in_test = set([d[0] for d in test])
  items_in_test = set([d[1] for d in test])
  users_in_valid = set([d[0] for d in valid])
  items_in_valid = set([d[1] for d in valid])
  users_in_train = set([d[0] for d in train])
  items_in_train = set([d[1] for d in train])

  print('Number of users / items in train: %d / %d' % (len(users_in_train),len(items_in_train)))
  print('Number of users / items in test: %d / %d' % (len(users_in_test),len(items_in_test)))
  print('Number of users / items in valid: %d / %d' % (len(users_in_valid),len(items_in_valid)))
  print('Number of users / items in test not in train: %d / %d' % 
          (len(users_in_test - users_in_train), len(items_in_test - items_in_train)))
  print('Number of users / items in valid not in train: %d / %d\n' % 
          (len(users_in_valid - users_in_train),len(items_in_valid - items_in_train)))

  return train, test, valid
