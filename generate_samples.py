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
seed = 700
embiggen_factor = 70


data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
initial_centroids = choose_random_centroids(samples, n_clusters,seed)
x=tf.Variable(initial_centroids,name="x")
# x=tf.identity(initial_centroids)
nearest_indices = assign_to_nearest(samples, x)
updated_centroids = update_centroids(samples, nearest_indices, n_clusters)

model = tf.initialize_all_variables()

with tf.Session() as session:
	session.run(model)
	sample_values = session.run(samples)
	initial_centroids_values = session.run(initial_centroids)
	# print(initial_centroids_values)
	y=initial_centroids_values
	assign_x=x.assign(y)
	session.run(assign_x)
	# print(x.eval())
	#x=tf.identity(initial_centroids)
	#updated_centroid_value = session.run(updated_centroids)

	for i in range(100):
		updated_centroid_value=session.run(updated_centroids)
		z=updated_centroid_value
		assign_x_new=x.assign(z)
		session.run(assign_x_new)
		print(x.eval())
		plot_clusters(sample_values,updated_centroid_value,n_samples_per_cluster)

	print(updated_centroid_value)

#plot_clusters(sample_values, updated_centroid_value, n_samples_per_cluster)