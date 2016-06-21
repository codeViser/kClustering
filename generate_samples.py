# import tensorflow as tf
# import numpy as np


# from functions import create_samples
# from functions import plot_clusters

# n_features = 2
# n_clusters = 3
# n_samples_per_cluster = 500
# seed = 700
# embiggen_factor = 70

# np.random.seed(seed)

# centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)

# model = tf.initialize_all_variables()
# with tf.Session() as session:
#     sample_values = session.run(samples)
#     centroid_values = session.run(centroids)
#     plot_clusters(sample_values, centroid_values, n_samples_per_cluster)

#==========================================================================


# import tensorflow as tf
# import numpy as np

# from functions import create_samples, choose_random_centroids, plot_clusters

# n_features = 2
# n_clusters = 3
# n_samples_per_cluster = 500
# seed = 700
# embiggen_factor = 70

# centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
# initial_centroids = choose_random_centroids(samples, n_clusters)

# model = tf.initialize_all_variables()
# with tf.Session() as session:
#     sample_values = session.run(samples)
#     updated_centroid_value = session.run(initial_centroids)

# plot_clusters(sample_values, updated_centroid_value, n_samples_per_cluster)

import tensorflow as tf
import numpy as np

from functions import *

n_features = 2
n_clusters = 3
n_samples_per_cluster = 500
seed = 700 #seed_max=4294967295-1
embiggen_factor = 70
threshold = 0.005


data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
initial_centroids = choose_random_centroids(samples, n_clusters,seed)
x=tf.Variable(initial_centroids,name="x")
nearest_indices = assign_to_nearest(samples, x)
updated_centroids = update_centroids(samples, nearest_indices, n_clusters)

model = tf.initialize_all_variables()

with tf.Session() as session:
	session.run(model)
	sample_values = session.run(samples)
	initial_centroids_values = session.run(initial_centroids)
	y=initial_centroids_values
	assign_x=x.assign(y)
	session.run(assign_x)

	j = threshold*1.0001

	while j > threshold:
		updated_centroid_value=session.run(updated_centroids)

		distances = tf.reduce_sum( tf.square(tf.sub(updated_centroids, x)), 1)
		maxDist = tf.argmax(distances,0)
		maxDist = tf.to_int32(maxDist)
		distances_value=distances.eval()
		j = distances_value[maxDist.eval()]

		print(j)

		z=updated_centroid_value
		assign_x_new=x.assign(z)
		session.run(assign_x_new)
		print(x.eval())

	#print(updated_centroid_value)

plot_clusters(sample_values, updated_centroid_value, n_samples_per_cluster)