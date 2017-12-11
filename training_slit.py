
import os, sys, glob
import random

def training_slit(in_dir, out_dir, percent):
    files = glob.glob(in_dir+"/*.jpg")
    random.shuffle(files)
    print(files)


if __name__ == '__main__':
    if len(sys.argv) > 3 :
        in_dir = sys.argv[1]
        out_dir = sys.argv[2]
        percent = sys.argv[3]

        training_slit(in_dir, out_dir, percent)

    else : print("Usage: training_slit <dir_in> <dir_out> <percent>")
