
import os
import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import argparse
import csv
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument("--sub_ts", default=128, type=int, help="Length of sub-time series intervals.")
parser.add_argument("--num_cl", default=10000, type=int, help="Number of clusters for sub-time series.")
parser.add_argument("--num_ts", default=13496, type=int, help="Number of time series to read.")
parser.add_argument("--name", default="ts_encoded.csv", type=str, help="Name of experiment.")
def pad(l, size):
    return l + [0 for i in range(size - len(l))]

def split_lst(l, sec):
    split_l = []
    n = int(np.floor(len(l)/float(sec)))
    for i in range(n):
        split_l.append(l[i*sec:(i+1)*sec])
    if len(l) > n*sec:
        split_l.append(pad(l[(n*sec):],sec))
    return split_l

def flatten(list_of_lists):
    flat = []
    for l in list_of_lists:
        for item in l:
            flat.append(item)
    return flat

def read_initial_data(path, num_ts, sub_ts):
    '''Args: path to FITS data files.
        Returns: a re-formatted time series, lengths, intervals
        '''
    count = 0
    rates = []
    lengths = []
    time_int = []
    max_rate = 0.
    min_rate = 1000
    #for filename in sorted(os.listdir(path))[:50]:
    for filename in sorted(os.listdir(path))[:num_ts]:
        count += 1
        if count % 1000 == 0:
            print "Reading timeseries:", count
        #print count, filename
        with fits.open(path+filename, memmap=False) as example:
            hdu = example[1]
            time = hdu.data.field("TIME")
            rate = hdu.data.field("RATE")
            #print "Rate length", len(rate)
            #error = hdu.data.field("ERROR")
            #print len(time), len(rate), len(error)
            max_rate = max(max_rate, max(rate))
            min_rate = min(min_rate, min(rate))
            avg_time = np.mean([time[i+1] - time[i] for i in range(len(time) -1)])
            time_int.append(avg_time)
            rates.append(rate)
            lengths.append(len(rate))
            del example
    print "Max rate:", max_rate, "Min rate:", min_rate
    # Split timeseries into 32-sec chuncks
    rates = [split_lst(list(rate), sub_ts) for rate in rates]
    # Normalize time series
    rates = [(np.array(rate)-min_rate)/max_rate for rate in rates]
    return rates, lengths, time_int

def main():
    # Read arguments
    FLAGS = parser.parse_args()
    # Read data
    data_path = "/work/kve216/blackhole_nlp/nicedata_for_daniela/"
    rates, lengths, time_int = read_initial_data(data_path, FLAGS.num_ts, FLAGS.sub_ts)
    print "Time series with avg time interval > 1:", len([ts for ts in time_int if ts > 1.1])
    mini_ts = flatten(rates)
    print "Number of mini timeseries:", len(mini_ts)
    # Run k-means on the chunks
    
    points_n = len(mini_ts)
    iteration_n = 100
    
    points = tf.constant(mini_ts)
    centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0, 0], [clusters_n, -1]))
    
    points_expanded = tf.expand_dims(points, 0)
    centroids_expanded = tf.expand_dims(centroids, 1)
    
    distances = tf.reduce_sum(tf.square(tf.sub(points_expanded, centroids_expanded)), )
    assignments = tf.argmin(distances, 0)
    
    means = []
    for c in xrange(clusters_n):
        means.append(tf.reduce_mean(tf.gather(points,tf.reshape(tf.where(tf.equal(assignments, c)),[1,-1])),reduction_indices=[1]))
    
    new_centroids = tf.concat(0, means)

    update_centroids = tf.assign(centroids, new_centroids)
    init = tf.initialize_all_variables()
    
    with tf.Session() as sess:
        sess.run(init)
        for step in xrange(iteration_n):
            [_, centroid_values, points_values, assignment_values] = sess.run([update_centroids, centroids, points, assignments])

    print "centroids" + "\n", centroid_values

#plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.5)
#plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)
#plt.show()


kmeans = KMeans(n_clusters=FLAGS.num_cl, random_state=0).fit(mini_ts)
    
    print "KMeans inertia:", kmeans.inertia_
    rates_enc = []
    for rate in rates:
        rate_enc = kmeans.predict(rate)
        rates_enc.append(rate_enc)
    # Save time series encodings
    with open(FLAGS.name, 'wb') as csvfile:
        writer = csv.writer(csvfile)
        for rate in rates_enc:
            writer.writerow(rate)
        csvfile.close()

if __name__ == '__main__':
    main()