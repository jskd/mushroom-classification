import cv2
import numpy as np
import os, sys

def find_uglies(directory, examples):

    for img in os.listdir(directory):
        for ugly in os.listdir(examples):
            try:
                current_image_path = str(directory)+'/'+str(img)
                ugly = cv2.imread(examples+"/"+str(ugly))
                question = cv2.imread(current_image_path)
                if ugly.shape == question.shape and not(np.bitwise_xor(ugly,question).any()):
                    print(current_image_path)
                    os.remove(current_image_path)
            except Exception as e:
                pass


if __name__ == '__main__':
    if len(sys.argv) > 1 :
        directory = sys.argv[1]
        examples = sys.argv[2]

        find_uglies(directory, examples)

    else : print("Usage: clean_up <dir> <dir_examples>")
