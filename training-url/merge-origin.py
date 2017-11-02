import urllib.request
import magic
import os, sys, glob

# form https://stackoverflow.com/questions/26819591/fastest-way-to-combine-several-text-files-without-duplicate-lines
def merge(patern, output):
    files = glob.glob(patern)
    all_lines = []
    for f in files:
        with open(f,'r') as fi:
            all_lines += fi.readlines()
    all_lines = set(all_lines)
    with open(output,'w') as fo:
        fo.write("".join(all_lines))

if __name__ == '__main__':
    merge('./mushroom/origin/*', './mushroom/merge.txt')


