"""
Code inspired from : https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/5_DataManagement/build_an_image_dataset.py
"""


import tensorflow as tf
import os
from datetime import datetime

DATASET_TRAIN = 'training_set/'
DATASET_TEST = 'test_set/'

# Paramètres des images
N_CLASSES = 17 
IMG_H = 32 
IMG_W = 32 
CHANNELS = 3

# Paramètres du model
batch_train_size = 64
batch_test_size = 64

learning_rate = 0.001
dropout = 0.75
num_steps = 20000
display_step = 100

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

def convolutional_network(x, n_classes, dropout, reuse, is_training):

	with tf.variable_scope('ConvNet', reuse=reuse):

		# Input Layer
		conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
		conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

		# Second layer
		conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
		conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

		# Fully connecter layer
		fc1 = tf.contrib.layers.flatten(conv2)
		fc1 = tf.layers.dense(fc1, 1024)
		fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

		# Output layer
		out = tf.layers.dense(fc1, n_classes)
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
		for step in range(1, num_steps+1):
			if step % display_step == 0:

				train, loss, t_acc, r_acc = sess.run([train_op, loss_op, accuracy_training, accuracy_test])

				print("  -> Step {} : loss({:.2f}%) t_accuracy({:.2f}%) r_accuracy({:.2f}%)".format(step, loss*100, t_acc*100, r_acc*100))

			else:
				sess.run(train_op)

	except Exception as e:
		print(e)
		coord.request_stop()
		coord.join(threads)

	coord.request_stop()
	coord.join(threads)