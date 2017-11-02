#!/bin/bash
cat origin/mushroom-*.txt > merge-duplicate.txt
sort -u merge-duplicate.txt > merge.txt
rm merge-duplicate.txt
