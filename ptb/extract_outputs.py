import tensorflow as tf
import numpy as np
import os
import time
import datetime
from tensorflow.contrib import learn
import blackholes_word_lm as bh
import reader




# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 1, "Batch Size (default: 1)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

config = get_config()
eval_config = get_config()
eval_config.batch_size = 1
eval_config.num_steps = 1

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
raw_data = reader.blackholes_raw_data("data/")
train_data, valid_data, test_data = raw_data
graph = tf.Graph()
with graph.as_default():
    with tf.name_scope("Test"):
      models = []
      test_sentences = bh.create_test_data(test_data)
          #with tf.variable_scope("Model", reuse=True, initializer=initializer):
      for sentence in test_sentences:
          test_input = bh.PTBInput(config=eval_config, data=sentence, name="TestInput")
          mtest = bh.PTBModel(is_training=False, config=eval_config,
                         input_= test_input)
          models.append((mtest,test_input))
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)


