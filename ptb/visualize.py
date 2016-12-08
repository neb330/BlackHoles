from ptb_word_lm import *
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
from scipy import spatial


parser = argparse.ArgumentParser()
parser.add_argument("-checkpoint_dir", "--checkpoint_dir", type=str, required=True)
parser.add_argument("-data_path", "--data_path", type=str, default= "simple-examples/data/")

args = parser.parse_args()    
data_path = args.data_path
checkpoint_dir = args.checkpoint_dir

#tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
#33tf.flags.DEFINE_string("data_path", "simple-examples/data/", "Path to training data")

train_path = os.path.join(data_path, "ptb.train.txt")
word_to_id = reader._build_vocab(train_path)
id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
#print(type(word_to_id))

def similarity():
    score = 0
    three_words = [['a','an','document'],['in','of','picture'],['nation','country','end'],['films','movies','almost'],['workers','employees','movies'],['institutions','organizations','big'],['assets','portfolio','down'],["'", ",",'quite'],['finance','acquisition','seems'],['good','great','minutes']]
    for words in three_words:
        try:
            index_1 = embedding[word_to_id[words[0]], :].eval()
            index_2 = embedding[word_to_id[words[1]], :].eval()
            index_3 = embedding[word_to_id[words[2]],:].eval()
            score += ( (1 - spatial.distance.cosine(index_1, index_2)) > (1 -spatial.distance.cosine(index_1, index_3)) )
        except:
            continue
    return score

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
  assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
  plt.figure(figsize=(18, 18))  #in inches
  for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatter(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')

  plt.savefig(filename)

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        #norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        #normalized_embeddings = embeddings / norm
        #valid_embeddings = tf.nn.embedding_lookup(
        #normalized_embeddings, valid_dataset)
        #similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        embedding = graph.get_operation_by_name("Model/embedding").outputs[0]

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 500
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
        normalized_embeddings = embedding / norm
        final_embeddings = normalized_embeddings.eval()
        #print(normalized_embeddings.get_shape())
        low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
        labels = [id_to_word[i] for i in range(plot_only)]
        plot_with_labels(low_dim_embs, labels)
        #s = similarity()
        #print(s)

        


