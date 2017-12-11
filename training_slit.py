
import os, sys, glob
import random

def training_slit(in_dir, out_dir, percent):
    files = glob.glob(in_dir+"/*.jpg")
    random.shuffle(files)

    test_set, training_set = [], []

    for i, f in enumerate(files):
        if i < len(files) * percent:
            test_set.append(f)
        else:
            training_set.append(f)

    print(len(test_set))
    print(len(training_set))


if __name__ == '__main__':
    if len(sys.argv) > 3 :
        in_dir = sys.argv[1]
        out_dir = sys.argv[2]
        percent = float( int(sys.argv[3]) / 100)
        training_slit(in_dir, out_dir, percent)

    else:
        print("Usage: training_slit <dir_in> <dir_out> <percent_in_training>")
