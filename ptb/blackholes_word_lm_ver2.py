# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.

Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329

There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.

The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size

The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:

$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz

To run:

$ python ptb_word_lm.py --data_path=simple-examples/data/

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import rnn_cell
import reader
import os
import reader_words
import csv

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None,
                    "Where the training/test data is stored.")

flags.DEFINE_string("data_type", None,
                    "Either black holes or words.")

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
checkpoints = os.path.abspath(os.path.join(os.path.curdir, out_dir, "checkpoints"))
flags.DEFINE_string("save_path", checkpoints,
                    "Model output directory.")

flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


class PTBInput(object):
  """The input data."""

  def __init__(self, config, data, name=None):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
    self.input_data, self.targets = reader_words.ptb_producer(data, batch_size, num_steps, name=name)



class PTBModel(object):
  """The PTB model."""

  def __init__(self, is_training, config, input_):
    self._input = input_

    batch_size = input_.batch_size
    num_steps = input_.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0, state_is_tuple=True)
    #lstm_cell = rnn_cell.GRUCell(size)
    if is_training and config.keep_prob < 1:
      lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers, state_is_tuple=True)

    self._initial_state = cell.zero_state(batch_size, data_type())

    with tf.device("/cpu:0"):
        self.embedding = tf.get_variable(
          "embedding", [vocab_size, size], dtype=data_type())
        inputs = tf.nn.embedding_lookup(self.embedding, input_.input_data)

    if is_training and config.keep_prob < 1:
      inputs = tf.nn.dropout(inputs, config.keep_prob)

    # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
    # This builds an unrolled LSTM for tutorial purposes only.
    # In general, use the rnn() or state_saving_rnn() from rnn.py.
    #
    # The alternative version of the code below is:
    #
    inputs = [tf.squeeze(input_step, [1])
               for input_step in tf.split(1, num_steps, inputs)]
    outputs, state = tf.nn.rnn(cell, inputs, initial_state=self._initial_state)
    
    
    '''outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)'''

    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    softmax_w = tf.get_variable(
        "softmax_w", [size, vocab_size], dtype=data_type())
    softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
    logits = tf.matmul(output, softmax_w) + softmax_b
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(input_.targets, [-1])],
        [tf.ones([batch_size * num_steps], dtype=data_type())])
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = state
    self._outputs = output

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=tf.contrib.framework.get_or_create_global_step())

    self._new_lr = tf.placeholder(
        tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def input(self):
    return self._input
  
  @property
  def outputs(self):
    return self._outputs
  
  
  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 0.1
  max_grad_norm = 5
  num_layers = 2
  num_steps = 100
  hidden_size = 500
  max_epoch = 15
  max_max_epoch = 15
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10002


class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 35
  hidden_size = 650
  max_epoch = 6
  max_max_epoch = 39
  keep_prob = 1.0
  lr_decay = 0.8
  batch_size = 10
  vocab_size = 10002


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 35
  hidden_size = 1500
  max_epoch = 14
  max_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15
  batch_size = 20
  vocab_size = 10002


class TestConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 1
  num_layers = 1
  num_steps = 2
  hidden_size = 2
  max_epoch = 1
  max_max_epoch = 1
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10000


def run_epoch(session, model, eval_op=None, verbose=False, is_testing = False):
  """Runs the model on the given data."""
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = session.run(model.initial_state)

  fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
      "outputs" : model.outputs
  }
  if eval_op is not None:
    fetches["eval_op"] = eval_op


  for step in range(model.input.epoch_size):
    feed_dict = {}
    for i, (c, h) in enumerate(model.initial_state):
      feed_dict[c] = state[i].c
      feed_dict[h] = state[i].h

    vals = session.run(fetches, feed_dict)
    cost = vals["cost"]
    states = vals["final_state"]
    outputs = vals["outputs"]

    costs += cost
    iters += model.input.num_steps

    if is_testing:
        print(outputs.shape)
        #final_outputs = outputs.mean(0)
        #arr[step,:] = final_outputs

    if verbose and step % (model.input.epoch_size // 10) == 10:
        print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
             iters * model.input.batch_size / (time.time() - start_time)))
  if is_testing:
    return np.exp(costs / iters), arr, states
  else:
    return np.exp(costs / iters), outputs, states



def create_test_data(data):
    data = [str(item) for item in data]
    sentences = [item for item in ' '.join(data).split('10001')]
    sentences = [words.split() for words in sentences]
    for (i,sentence) in enumerate(sentences):
        if len(sentence) < 2:
            del sentences[i]
    for (i,sentence) in enumerate(sentences):
        for (j,word) in enumerate(sentence):
            sentences[i][j] = int(word)
    return sentences



def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  elif FLAGS.model == "medium":
    return MediumConfig()
  elif FLAGS.model == "large":
    return LargeConfig()
  elif FLAGS.model == "test":
    return TestConfig()
  else:
    raise ValueError("Invalid model: %s", FLAGS.model)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")
  
  if FLAGS.data_type == 'black_holes':
    raw_data = reader.blackholes_raw_data(FLAGS.data_path)
  elif FLAGS.data_type == 'words':
    raw_data = reader_words.ptb_raw_data(FLAGS.data_path)
  else:
    print("Enter a correct data type -- either black_holes or words.")

  train_data, valid_data, test_data = raw_data

  config = get_config()
  eval_config = get_config()
  eval_config.batch_size = 1
  eval_config.num_steps = 100
  graph = tf.Graph()
  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    #saver = tf.train.Saver(tf.all_variables())
    with tf.name_scope("Train"):
      train_input = PTBInput(config=config, data=train_data, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True, config=config, input_=train_input)
      tf.scalar_summary("Training Loss", m.cost)
      tf.scalar_summary("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = PTBInput(config=config, data=valid_data, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False, config=config, input_=valid_input)
      tf.scalar_summary("Validation Loss", mvalid.cost)


    with tf.name_scope("Test"):
      test_input = PTBInput(config=eval_config, data=test_data, name=("TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = PTBModel(is_training=False, config=eval_config, input_= test_input)
                            

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
  
      for i in range(config.max_max_epoch):
          
        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity, _, _ = run_epoch(session, m, eval_op=m.train_op,
                                     verbose=True)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity, _, _ = run_epoch(session, mvalid)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
      
       
      test_perplexity, arr, states = run_epoch(session, mtest, is_testing= True)
      print("Test Perplexity: %.3f" % test_perplexity)
    
      if FLAGS.data_type == 'black_holes':
        print("Saving outputs (%d,%d) to black_hole_outputs.csv" % (arr.shape[0],arr.shape[1]))
        np.savetxt('black_hole_outputs.csv', arr, delimiter=',')
      elif FLAGS.data_type == 'words':
        print("Saving outputs (%d,%d) to word_outputs.csv" % (arr.shape[0],arr.shape[1]))
        np.savetxt('word_outputs.csv', arr, delimiter=',')


      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
  tf.app.run()