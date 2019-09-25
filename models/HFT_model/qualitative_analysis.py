import argparse, os, csv
import pickle

import numpy as np

def parse_args():

  parser = argparse.ArgumentParser(description='Qualitative analysis of HFT model')

  parser.add_argument('--input_file_path', action='store', default='',
                      help='path to input files')

  parser.add_argument('--top_n', action="store", type=int, default=10,
                      help='number of words to show per topic.')
  
  return parser.parse_args()

def show_top_n_words_for_topics(n, vocab, Psi):

  # Normalize (subtract mean) by column (words in vocab)
  b_W = np.mean(Psi, axis=0)
  
  offset = Psi - b_W
  
  # For each topic (row), show top 10 words based on asbolute value
  num_topic = Psi.shape[0]
  for k in range(num_topic):
    print('Topic %d\n' % (k+1))
    score_sorted = sorted([(idx, score) for idx, score in enumerate(offset[k,])], 
                          key=lambda x:abs(x[1]), reverse=True)
    top_n_words = [vocab[s[0]] for s in score_sorted[:n]]
    print('%s\n' % ', '.join(top_n_words))

def load_data(path):

  vocab = {}
  reader = csv.reader(open(os.path.join(path, 'vocab.csv')), quotechar='|')
  for row in reader:
    vocab[int(row[0])] = row[1]

  with open(os.path.join(path, 'hft_model.pickle'), 'rb') as f:
    kappa, Theta, Psi, Phi = pickle.load(f)
  
  print('kappa: %.4f\n' % kappa)
  return vocab, kappa, Theta, Psi, Phi

def main(args):

  vocab, kappa, Theta, Psi, Phi = load_data(args.input_file_path)
  show_top_n_words_for_topics(args.top_n, vocab, Psi)

if __name__ == '__main__':

  # parse arguments
  args = parse_args()
  print("="*50)
  for arg in vars(args):
    print("  %s: %s" % (arg, str(getattr(args, arg))))
  print("="*50+"\n")
  main(args)
  