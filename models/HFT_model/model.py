from datetime import datetime
import time

import numpy as np
import tensorflow as tf

# Offset model
class offsetModel(object):

  def __init__(self, train_ratings, valid_ratings, test_ratings):
    print('\nBuilding offset model...')
    self.train_ratings = train_ratings
    self.valid_ratings = valid_ratings
    self.test_ratings = test_ratings

  def train(self):
    self.alpha = np.mean([r[2] for r in self.train_ratings])
    print('Offset: %.4f' % self.alpha)

    self.train_mse = np.mean([(r[2]-self.alpha)**2 for r in self.train_ratings])
    self.valid_mse = np.mean([(r[2]-self.alpha)**2 for r in self.valid_ratings])
    self.test_mse = np.mean([(r[2]-self.alpha)**2 for r in self.test_ratings])
    print('MSE - training: %.4f, valid: %.4f, test: %.4f\n' % 
          (self.train_mse, self.valid_mse, self.test_mse))

# Hidden Factors as Topcis model
class HFTModel(object):

  def __init__(self, hps, train_ratings, valid_ratings, test_ratings,
               train_item_doc, valid_item_doc,
               num_users, num_items, num_reviews, log_folder):
    self._session = None
    self._hps = hps
    self.train_ratings = train_ratings
    self.valid_ratings = valid_ratings
    self.test_ratings  = test_ratings
    self.train_item_doc = train_item_doc
    self.valid_item_doc = valid_item_doc
    self.max_item_len = max([len(d) for d in self.train_item_doc])
    self.num_users = num_users
    self.num_items = num_items
    self.num_reviews = num_reviews
    self.log_folder = log_folder

  def _create_variables(self):

    hps = self._hps

    tf.compat.v1.random.set_random_seed(1)

    ##############################################################################
    # variables for rating loss
    with tf.compat.v1.variable_scope('rating_loss'):  

      self.A = tf.SparseTensor(
          indices=[r[:2] for r in self.train_ratings],
          values=[r[2] for r in self.train_ratings],
          dense_shape=[self.num_users, self.num_items])

      self.A_valid = tf.SparseTensor(
          indices=[r[:2] for r in self.valid_ratings],
          values=[r[2] for r in self.valid_ratings],
          dense_shape=[self.num_users, self.num_items])

      self.A_test = tf.SparseTensor(
          indices=[r[:2] for r in self.test_ratings],
          values=[r[2] for r in self.test_ratings],
          dense_shape=[self.num_users, self.num_items])
      
      # overall bias
      train_offset = np.mean([r[2] for r in self.train_ratings])
      self.alpha = tf.Variable([train_offset])
      # Initialize the embeddings using a normal distribution.
      self.U = tf.Variable(tf.random.normal(
          [self.num_users, hps.emb_dim], stddev=hps.init_stddev), name='user_embedding')
      self.user_bias = tf.Variable(tf.random.normal(
          [self.num_users, 1], stddev=hps.init_stddev), name='user_bias')
      self.U_plus_bias = tf.concat([self.U, self.user_bias, tf.ones((self.num_users,1))], 
                                    axis=1)

      self.V = tf.Variable(tf.random.normal(
          [self.num_items, hps.emb_dim], stddev=hps.init_stddev), name='item_embedding')
      self.item_bias = tf.Variable(tf.random.normal(
          [self.num_items, 1], stddev=hps.init_stddev), name='item_bias')
      self.V_plus_bias = tf.concat([self.V, tf.ones((self.num_items,1)), self.item_bias], 
                                    axis=1)

    with tf.compat.v1.variable_scope('doc_ll'):

      # sampled word topic
      self.z = tf.Variable(tf.random.uniform([len(self.train_item_doc), self.max_item_len], 
                                              minval=0, maxval=hps.emb_dim,
                                              dtype=tf.int64), trainable=False, 
                                              name='sampled_word_topic')
      #self.z = tf.assign(self.z, f(W,resid))
      self.kappa = tf.Variable(tf.random.uniform([1], minval=hps.min_kappa, 
                                                      maxval=hps.max_kappa), name='kappa')
      self.Theta = tf.nn.softmax(self.kappa*self.V, axis=1, name='topic_dist_per_item')
      self.Psi = tf.Variable(tf.random.normal([hps.emb_dim, hps.vocab_size], 
                                               stddev=hps.init_stddev))
      self.Phi = tf.nn.softmax(self.Psi, axis=1, name='word_dist_per_topic')

  def _sample_z(self):

    z_sample = []
    for idx_doc in range(len(self.train_item_doc)):

      #print(len(self.train_item_doc[idx_doc]))
      doc_topic_dist = self.Theta[idx_doc,:]
      #print(doc_topic_dist.get_shape())

      topic_word_dist = tf.gather(self.Phi, self.train_item_doc[idx_doc], axis=1)
      #print(topic_word_dist.get_shape())

      topic_dist = tf.multiply(tf.expand_dims(doc_topic_dist,-1), topic_word_dist)
      #print(topic_dist.get_shape())

      doc_z_sample = tf.random.categorical(tf.transpose(topic_dist), num_samples=1)
      #print(doc_z_sample.get_shape())

      doc_z_sample_padded = tf.pad(tf.reshape(doc_z_sample,[-1]), 
                                  [[0, self.max_item_len-len(self.train_item_doc[idx_doc])]],
                                  'CONSTANT')
      #print(doc_z_sample_padded.get_shape())
      z_sample.append(doc_z_sample_padded)
      
    #print(tf.stack(z_sample).get_shape())
    self.z = tf.compat.v1.assign(self.z, tf.stack(z_sample))

  def _corpus_likelihood(self):
    ll_all_docs = []
    for idx_doc in range(len(self.train_item_doc)):

      doc_len = len(self.train_item_doc[idx_doc])
      #print(doc_len)
      # probability of topic for each words in document (1 x # of words)
      idx_topic = self.z[idx_doc,:doc_len]
      #print(idx_topic.get_shape())
      # select the probability of topic in a given item
      theta_doc = tf.gather(self.Theta[idx_doc,:], idx_topic)
      #print(theta_doc.get_shape())      
      
      # probability of each word being assigned to the topic (1 x # of words)
      idx_topic_word = tf.stack([self.z[idx_doc,:doc_len], self.train_item_doc[idx_doc]], 
                                  axis=1)
      #print(idx_topic_word.get_shape())
      phi_topic_word = tf.gather_nd(self.Phi, idx_topic_word)
      #print(phi_topic_word.get_shape())

      #print((theta_doc*phi_topic_word).get_shape())
      ll_one_doc = tf.math.reduce_sum(theta_doc*phi_topic_word)
      ll_all_docs.append(ll_one_doc)
      
    #print(tf.stack(ll_all_docs).get_shape())
    self.ll_docs = tf.math.reduce_sum(tf.stack(ll_all_docs))
    tf.summary.scalar('corpus_likelihood', self.ll_docs)

  def _inference(self):

    pred_train = tf.reduce_sum(
      tf.gather(self.U_plus_bias, self.A.indices[:, 0]) *
      tf.gather(self.V_plus_bias, self.A.indices[:, 1]), axis=1) + \
      tf.tile(self.alpha, [len(self.train_ratings)])
    
    self.train_mse = tf.reduce_mean(tf.square(tf.subtract(self.A.values, pred_train)))
    self.rating_loss = tf.compat.v1.losses.mean_squared_error(self.A.values, pred_train)

    pred_valid = tf.reduce_sum(
      tf.gather(self.U_plus_bias, self.A_valid.indices[:, 0]) *
      tf.gather(self.V_plus_bias, self.A_valid.indices[:, 1]), axis=1) + \
      tf.tile(self.alpha, [len(self.valid_ratings)])
    
    self.valid_mse = tf.reduce_mean(tf.square(tf.subtract(self.A_valid.values, 
                                                          pred_valid)))

    pred_test = tf.reduce_sum(
      tf.gather(self.U_plus_bias, self.A_test.indices[:, 0]) *
      tf.gather(self.V_plus_bias, self.A_test.indices[:, 1]), axis=1) + \
      tf.tile(self.alpha, [len(self.test_ratings)])
    
    self.test_mse = tf.reduce_mean(tf.square(tf.subtract(self.A_test.values, 
                                                         pred_test)))

    tf.summary.scalar('mse_train', self.train_mse)
    tf.summary.scalar('mse_valid', self.valid_mse)
    tf.summary.scalar('mse_test', self.test_mse)
    tf.summary.scalar('rating_error', self.rating_loss)
    
  def _add_train_ops(self):

    hps = self._hps
    # overall loss
    self.total_loss = self.rating_loss - hps.mu*self.ll_docs
    tf.summary.scalar('total_loss', self.total_loss)

    # Optimize variables with fixing z (sampled word topic)
    self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.total_loss, 
                                                            method='L-BFGS-B',
                                                            var_to_bounds={self.kappa:(0, np.infty)},
                                                            options={'maxiter': hps.max_iter_steps})

  def build_graph(self):
    tf.compat.v1.logging.info('Building graph...')
    t0 = time.time()
    self._create_variables()
    self._corpus_likelihood()
    self._inference()
    self._add_train_ops()
    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self._summaries = tf.compat.v1.summary.merge_all()
    t1 = time.time()
    tf.compat.v1.logging.info('Time to build graph: %i seconds', t1 - t0)

  def train(self):
    # train for 2500 iterations
    # at every 50 iterations, topic assignments are updated and MSE is reported.
    hps = self._hps
    
    with tf.compat.v1.Session() as sess:

      self.summary_writer = tf.compat.v1.summary.FileWriter(self.log_folder,
                                                            sess.graph)

      sess.run(tf.compat.v1.global_variables_initializer())      
      Theta_prev, Phi_prev = sess.run([self.Theta, self.Phi])

      for step in range(hps.num_iter_steps):
        self.optimizer.minimize(sess)
        
        rating_loss, train_mse, valid_mse, test_mse, \
        ll_docs, total_loss, Theta, Phi, summary = \
          sess.run([self.rating_loss, self.train_mse, self.valid_mse, self.test_mse, 
                    self.ll_docs, self.total_loss, self.Theta, self.Phi,
                    self._summaries])
        print('Step: %i - Rating loss: %.4f, Corpus likelihood: %.4f' % 
              (step, rating_loss, ll_docs))
        print('MSE - training: %.4f, valid: %.4f, test: %.4f\n' % 
              (train_mse, valid_mse, test_mse))

        # check the difference of Theta, Phi before and after the optimization
        e_Theta = np.linalg.norm(Theta - Theta_prev)
        e_Phi = np.linalg.norm(Phi - Phi_prev)

        self.summary_writer.add_summary(summary, step)

        # terminate if Theta and Phi converged
        if e_Theta < hps.threshold and e_Phi < hps.threshold:
          break
        
        Theta_prev = Theta
        Phi_prev = Phi

        self._sample_z()
        