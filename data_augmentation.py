import cv2
import numpy as np
import os, math

DIR_PATH = "training_set/"

nb_images = 0

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

def change_gamma(img, ratio_gamma):
	invGamma = 1.0 / ratio_gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	return cv2.LUT(img, table)

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

		if not image_name.startswith(("g6_", "g4_", "g2_", "g05_", "g025_")):
			img = cv2.imread(link,1)

			#dst = change_gamma(img, 6)
			#cv2.imwrite("{}/g6_{}".format(image_path, image_name), dst)

			dst = change_gamma(img, 4)
			cv2.imwrite("{}/g4_{}".format(image_path, image_name), dst)

			#dst = change_gamma(img, 2)
			#cv2.imwrite("{}/g2_{}".format(image_path, image_name), dst)

			dst = change_gamma(img, 0.5)
			cv2.imwrite("{}/g05_{}".format(image_path, image_name), dst)

			#dst = change_gamma(img, 0.25)
			#cv2.imwrite("{}/g025_{}".format(image_path, image_name), dst)

def blur(links_liste):
	for link in links_liste:
		image_name = link.split("/")[-1]
		image_path = os.path.dirname(link)

		if not image_name.startswith(("b_")):
			img = cv2.imread(link,1)
			dst = cv2.blur(img,(6,6))
			cv2.imwrite("{}/b_{}".format(image_path, image_name), dst)

def zoom(links_liste):
	for link in links_liste:
		image_name = link.split("/")[-1]
		image_path = os.path.dirname(link)

		if not image_name.startswith(("z_")):
			img = cv2.imread(link,1)
			height, width = img.shape[:2]
			image_center = (width / 2, height / 2)

			ratio = 1.4
			img_scaled = cv2.resize(img, (0,0), fx=ratio, fy=ratio)

			offset_h = int(((height*ratio) - height)/2)
			offset_w = int(((width*ratio) - width)/2)

			dst = img_scaled[offset_h:(height+offset_h), offset_w:(width+offset_w)]
			cv2.imwrite("{}/z_{}".format(image_path, image_name), dst)


def perspective(links_liste):
	for link in links_liste:
		image_name = link.split("/")[-1]
		image_path = os.path.dirname(link)

		if not image_name.startswith(("p_")):
			img = cv2.imread(link,1)
			height, width = img.shape[:2]
			image_center = (width / 2, height / 2)

			pts1 = np.float32([[50,50],[width-100,100],[77,height-80],[width,height]])
			pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])

			M = cv2.getPerspectiveTransform(pts1,pts2)
			dst = cv2.warpPerspective(img,M,(width, height))

			cv2.imwrite("{}/p_{}".format(image_path, image_name), dst)

def noise(links_liste):
	for link in links_liste:
		image_name = link.split("/")[-1]
		image_path = os.path.dirname(link)

		if not image_name.startswith(("nd_", "nl_")):
			img = cv2.imread(link,1)
			row,col,ch= img.shape
			var = 100
			sigma = var**0.5

			gauss = np.random.normal(80, sigma,(row,col,ch))
			gauss = gauss.reshape(row,col,ch)
			dst = img + gauss
			cv2.imwrite("{}/nl_{}".format(image_path, image_name), dst)

			gauss = np.random.normal(-60, sigma,(row,col,ch))
			gauss = gauss.reshape(row,col,ch)
			dst = img + gauss
			cv2.imwrite("{}/nd_{}".format(image_path, image_name), dst)



images_links = get_images_link(DIR_PATH)

#print("# Change perspective ...")
#perspective(images_links)

print("# Zooming images ...")
zoom(images_links)

print("# Bluring images ...")
blur(images_links)

#images_links = get_images_link(DIR_PATH)
print("# Add noise to images ...")
noise(images_links)

#images_links = get_images_link(DIR_PATH)
#print("# Flipping images ...")
#apply_flip(images_links)

#print("# Rotating images ...")
#apply_rotations(images_links)
