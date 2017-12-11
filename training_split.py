
import os, sys, glob
import random
import ntpath
from shutil import copyfile
from glob import glob

def training_split(in_dir, out_dir, percent):

    TRAINING_DIR = out_dir + "/training_set/"
    TEST_DIR = out_dir + "/test_set/"
    PATTERN = "*.jpg"
    files = []
    test_set, training_set = [], []

    os.makedirs(TRAINING_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    for dir,_,_ in os.walk(in_dir):
        files.extend(glob(os.path.join(dir, PATTERN)))

    random.shuffle(files)

    for i, f in enumerate(files):
        if i < len(files) * percent:
            test_set.append(f)
        else:
            training_set.append(f)

    for i, f in enumerate(test_set):
        #copyfile(f, TEST_DIR +ntpath.basename(f) )
        os.rename(f, TEST_DIR +ntpath.basename(f) )

    for i, f in enumerate(training_set):
        #copyfile(f, TRAINING_DIR +ntpath.basename(f) )
        os.rename(f, TRAINING_DIR +ntpath.basename(f) )

if __name__ == '__main__':
    if len(sys.argv) > 3 :
        in_dir = sys.argv[1]
        out_dir = sys.argv[2]
        percent = float( int(sys.argv[3]) / 100)
        training_split(in_dir, out_dir, percent)

    else:
        print("Usage: training_slit <dir_in> <dir_out> <percent_in_training>")
