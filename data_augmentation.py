import cv2
import numpy as np
import os

DIR_PATH = "training_set/"

list_dir = sorted(os.walk(DIR_PATH).__next__()[1])

nb_images = 0

for d in list_dir:
	c_dir = os.path.join(DIR_PATH, d)
	walk = os.walk(c_dir).__next__()

	for e in walk[2]:
		if e.endswith('.jpg') or e.endswith('.jpeg'):
			image_link = os.path.join(c_dir, e)
			image_name = image_link.split("/")[-1]

			if not image_name.startswith('r_') \
					and not image_name.startswith('f_') \
					 and not image_name.startswith('rf_'):

				print("# Augmentation of {}".format(image_name))
				nb_images += 1

				img = cv2.imread(image_link,1)
				rows,cols, _ = img.shape

				# Rotate 180
				M = cv2.getRotationMatrix2D((cols/2,rows/2),180,1)
				dst = cv2.warpAffine(img,M,(cols,rows))
				cv2.imwrite("{}/r_{}".format(c_dir, image_name), dst)

				# Flip vertical
				dst = cv2.flip(img,1)
				cv2.imwrite("{}/f_{}".format(c_dir, image_name), dst)

				# Rotate 180 and flip vertical
				dst = cv2.warpAffine(img,M,(cols,rows))
				dst = cv2.flip(dst,1)
				cv2.imwrite("{}/rf_{}".format(c_dir, image_name), dst)

print("Done. {} -> {} images".format(nb_images, nb_images*4))
