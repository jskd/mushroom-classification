import cv2
import numpy as np
import os, math

DIR_PATH = "training_set/"

nb_images = 0

def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def get_images_link(path):
	links = []
	list_dir = sorted(os.walk(path).__next__()[1])

	for d in list_dir:
		c_dir = os.path.join(path, d)
		walk = os.walk(c_dir).__next__()

		for e in walk[2]:
			if e.endswith('.jpg') or e.endswith('.jpeg'):
				links.append(os.path.join(c_dir, e))
	return links
					
def apply_rotations(links_liste):
	for link in links_liste:
		image_name = link.split("/")[-1]
		image_path = os.path.dirname(link)

		if not image_name.startswith(("r90_", "r180_", "r270_")):
			img = cv2.imread(link,1)
			# Rotate 90
			dst = rotate_image(img, 90)
			cv2.imwrite("{}/r90_{}".format(image_path, image_name), dst)

			# Rotate 180
			dst = rotate_image(img, 180)
			cv2.imwrite("{}/r180_{}".format(image_path, image_name), dst)

			# Rotate 270
			dst = rotate_image(img, 270)
			cv2.imwrite("{}/r270_{}".format(image_path, image_name), dst)

def apply_flip(links_liste):
	for link in links_liste:
		image_name = link.split("/")[-1]
		image_path = os.path.dirname(link)

		if not image_name.startswith(("fh_", "fv_")):
			img = cv2.imread(link,1)
			# Flip vertical
			dst = cv2.flip(img,1)
			cv2.imwrite("{}/fv_{}".format(image_path, image_name), dst)

			# Flip horizontal
			dst = cv2.flip(img,0)
			cv2.imwrite("{}/fh_{}".format(image_path, image_name), dst)

def change_brightness(links_liste):
	for link in links_liste:
		image_name = link.split("/")[-1]
		image_path = os.path.dirname(link)

		if not image_name.startswith(("l_", "d_")):
			img = cv2.imread(link,1)

			# light
			invGamma = 1.0 / 2
			table = np.array([((i / 255.0) ** invGamma) * 255
				for i in np.arange(0, 256)]).astype("uint8")
			dst = cv2.LUT(img, table)
			cv2.imwrite("{}/l_{}".format(image_path, image_name), dst)

			# dark
			invGamma = 1.0 / 0.5
			table = np.array([((i / 255.0) ** invGamma) * 255
				for i in np.arange(0, 256)]).astype("uint8")
			dst = cv2.LUT(img, table)
			cv2.imwrite("{}/d_{}".format(image_path, image_name), dst)
			

print("# Getting images links ...")
images_links = get_images_link(DIR_PATH)
print("# Change brightness ...")
change_brightness(images_links)
print("# Getting images links ...")
images_links = get_images_link(DIR_PATH)
print("# Flipping images ...")
apply_flip(images_links)
print("# Rotating images ...")
apply_rotations(images_links)
