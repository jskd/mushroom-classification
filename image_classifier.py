"""
Code inspired from : https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_DataManagement/build_an_image_dataset.py
"""

import tensorflow as tf
import os
from datetime import datetime
import time

DATASET_TRAIN = 'training_set/'
DATASET_TEST = 'test_set/'

# Paramètres des images
N_CLASSES = 17
IMG_H = 64
IMG_W = 64
CHANNELS = 3
batch_train_size = 128
batch_test_size = 128

def read_images(path, batch_size):
	images, labels = [], []
	label = 0
	# Lecture des différentes classes (dossiers)
	classes = sorted(os.walk(path).__next__()[1])

	for c in classes:
		c_dir = os.path.join(path, c)
		walk = os.walk(c_dir).__next__()

		for e in walk[2]:
			if e.endswith('.jpg') or e.endswith('.jpeg'):
				images.append(os.path.join(c_dir, e))
				labels.append(label)
		label += 1

	images = tf.convert_to_tensor(images, dtype=tf.string)
	labels = tf.convert_to_tensor(labels, dtype=tf.int32)
	image, label = tf.train.slice_input_producer([images, labels], shuffle=True)

	image = tf.read_file(image)
	image = tf.image.decode_jpeg(image, channels=CHANNELS)
	# Resize images
	image = tf.image.resize_images(image, [IMG_H, IMG_W])
	# Normalize
	image = image * 1.0/127.5 - 1.0
	# Batching
	X, Y = tf.train.batch([image, label], batch_size=batch_size, capacity=100000, allow_smaller_final_batch=True, num_threads=4)
	return X, Y

# Images training
X, Y = read_images(DATASET_TRAIN, batch_train_size)
# Images testing
X_, Y_ = read_images(DATASET_TEST, batch_test_size)

# Paramètres du model
learning_rate = 0.001
dropout = 0.75
num_steps = 50000
display_step = 100

def convolutional_network(x, n_classes, dropout, reuse, is_training):
	with tf.variable_scope('ConvNet', reuse=reuse):

		"""
		# VGG 16
		c1 = tf.layers.conv2d(x, 64, 3, strides=(1, 1), activation=tf.nn.relu, padding="SAME")
		c2 = tf.layers.conv2d(c1, 64, 3, strides=(1, 1), activation=tf.nn.relu, padding="SAME")
		c2 = tf.layers.max_pooling2d(c2, 2, 2, padding="SAME")

		c3 = tf.layers.conv2d(c2, 128, 3, strides=(1, 1), activation=tf.nn.relu, padding="SAME")
		c4 = tf.layers.conv2d(c3, 128, 3, strides=(1, 1), activation=tf.nn.relu, padding="SAME")
		c4 = tf.layers.max_pooling2d(c4, 2, 2, padding="SAME")

		c5 = tf.layers.conv2d(c4, 256, 3, strides=(1, 1), activation=tf.nn.relu, padding="SAME")
		c6 = tf.layers.conv2d(c5, 256, 3, strides=(1, 1), activation=tf.nn.relu, padding="SAME")
		c7 = tf.layers.conv2d(c6, 256, 3, strides=(1, 1), activation=tf.nn.relu, padding="SAME")
		c7 = tf.layers.max_pooling2d(c7, 2, 2, padding="SAME")

		c8 = tf.layers.conv2d(c7, 512, 3, strides=(1, 1), activation=tf.nn.relu, padding="SAME")
		c9 = tf.layers.conv2d(c8, 512, 3, strides=(1, 1), activation=tf.nn.relu, padding="SAME")
		c10 = tf.layers.conv2d(c9, 512, 3, strides=(1, 1), activation=tf.nn.relu, padding="SAME")
		c10 = tf.layers.max_pooling2d(c10, 2, 2, padding="SAME")

		c11 = tf.layers.conv2d(c10, 512, 3, strides=(1, 1), activation=tf.nn.relu, padding="SAME")
		c12 = tf.layers.conv2d(c11, 512, 3, strides=(1, 1), activation=tf.nn.relu, padding="SAME")
		c13 = tf.layers.conv2d(c12, 512, 3, strides=(1, 1), activation=tf.nn.relu, padding="SAME")
		c13 = tf.layers.max_pooling2d(c13, 2, 2, padding="SAME")

		fc14 = tf.contrib.layers.flatten(c13)
		# Fully connecter layer
		fc14 = tf.layers.dense(fc14, 4096, activation=tf.nn.relu)
		fc14 = tf.layers.dropout(fc14, rate=dropout, training=is_training)

		fc15 = tf.layers.dense(fc14, 4096, activation=tf.nn.relu)
		fc15 = tf.layers.dropout(fc15, rate=dropout, training=is_training)

		# Output layer
		out = tf.layers.dense(fc15, n_classes)
		if not is_training : out = tf.nn.softmax(out)
		"""
		"""
		#ALEXNET
		c1 = tf.layers.conv2d(x, 96, 11, strides=(4, 4), activation=tf.nn.relu, padding="SAME")
		c1 = tf.layers.max_pooling2d(c1, 2, 2, padding="SAME")

		c2 = tf.layers.conv2d(c1, 256, 5, strides=(1, 1), activation=tf.nn.relu, padding="SAME")
		c2 = tf.layers.max_pooling2d(c2, 2, 2, padding="SAME")

		c3 = tf.layers.conv2d(c2, 384, 3, strides=(1, 1), activation=tf.nn.relu, padding="SAME")

		c4 = tf.layers.conv2d(c3, 384, 3, strides=(1, 1), activation=tf.nn.relu, padding="SAME")

		c5 = tf.layers.conv2d(c4, 256, 3, strides=(1, 1), activation=tf.nn.relu, padding="SAME")
		#c5 = tf.layers.max_pooling2d(c5, 2, 2, padding="SAME")

		fc6 = tf.contrib.layers.flatten(c5)

		fc7 = tf.layers.dense(fc6, 4096, activation=tf.nn.relu)
		fc7 = tf.layers.dropout(fc7, rate=dropout, training=is_training)

		fc8 = tf.layers.dense(fc7, 4096, activation=tf.nn.relu)
		fc8 = tf.layers.dropout(fc8, rate=dropout, training=is_training)

		# Output layer
		out = tf.layers.dense(fc8, n_classes)
		if not is_training : out = tf.nn.softmax(out)
		"""

		# Input Layer
		conv1 = tf.layers.conv2d(x, 32, 5, strides=(1, 1), activation=tf.nn.relu, padding="SAME")
		conv1 = tf.layers.max_pooling2d(conv1, 2, 2, padding="SAME")

		conv2 = tf.layers.conv2d(conv1, 64, 3, strides=(1, 1), activation=tf.nn.relu, padding="SAME")
		conv2 = tf.layers.max_pooling2d(conv2, 2, 2, padding="SAME")

		conv3 = tf.layers.conv2d(conv2, 128, 3, strides=(1, 1), activation=tf.nn.relu, padding="SAME")
		conv3 = tf.layers.max_pooling2d(conv3, 2, 2, padding="SAME")

		fc4 = tf.contrib.layers.flatten(conv3)
		# Fully connecter layer
		fc4 = tf.layers.dense(fc4, 1024, activation=tf.nn.relu)
		fc4 = tf.layers.dropout(fc4, rate=dropout, training=is_training)

		fc5 = tf.layers.dense(fc4, 512, activation=tf.nn.relu)
		fc5 = tf.layers.dropout(fc5, rate=(dropout-0.25), training=is_training)

		# Output layer
		out = tf.layers.dense(fc5, n_classes)
		if not is_training : out = tf.nn.softmax(out)


	return out

# Create training graph
logits_train = convolutional_network(X, N_CLASSES, dropout, reuse=False, is_training=True)
# Create training test graph
logits_train_test = convolutional_network(X, N_CLASSES, dropout, reuse=True, is_training=False)
# Create testing graph
logits_test = convolutional_network(X_, N_CLASSES, dropout, reuse=True, is_training=False)

# Optimizer
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluation training
correct_pred_train = tf.equal(tf.argmax(logits_train_test, 1), tf.cast(Y, tf.int64))
accuracy_training = tf.reduce_mean(tf.cast(correct_pred_train, tf.float32))

# Evaluation testing
correct_pred_test = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y_, tf.int64))
accuracy_test = tf.reduce_mean(tf.cast(correct_pred_test, tf.float32))

# Initialisation
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess, coord)

	# Training
	print("#######################################################################")
	print("# CLASS [{}] | IMG_SIZE [{},{}] | RATE [{}] | DROPOUT [{}]".format(N_CLASSES, IMG_W, IMG_H, learning_rate, dropout))
	print("#######################################################################")
	print("# Training ...")

	try:
		time_begin = time.time()
		for step in range(1, num_steps+1):
			if step % display_step == 0:

				time_end = time.time()

				train, loss, t_acc, r_acc = sess.run([train_op, loss_op, accuracy_training, accuracy_test])
				print("  -> Step {} : loss({:.2f}%) t_accuracy({:.2f}%) r_accuracy({:.2f}%) in {:.2f} min".format(
					step, loss*100, t_acc*100, r_acc*100, ((time_end - time_begin) / 60) ))

				time_begin = time.time()

			else:
				sess.run(train_op)

	except Exception as e:
		print(e)
		coord.request_stop()
		coord.join(threads)

	coord.request_stop()
	coord.join(threads)
