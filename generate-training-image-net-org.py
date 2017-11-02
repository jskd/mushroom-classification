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


def store_raw_images(directory, urlfile):
    pic_num = 0

    if not os.path.exists(directory):
        os.makedirs(directory)

    line = 1
    with open(urlfile) as f:
        for i in f:
            try:
                info = "(line:" + str(line).zfill(4) + ", pic-num:" + str(pic_num).zfill(4) + ", url:" + i.rstrip() + ")"
                address = urllib.request.urlopen(i, timeout=1)
                if(address != None):
                    filename = directory+"/"+str(pic_num)+".jpg"
                    urllib.request.urlretrieve(i, filename)
                    if(magic.from_file( filename, mime=True ) == "image/jpeg"):
                        print("[PASS] Image save " + info)
                        pic_num += 1
                    else:
                        print("[FAIL] Not jpeg   " + info)

            except Exception as e:
                print("[FAIL] Exception  " + info)
                pass
                line += 1


if __name__ == '__main__':
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        training_set_root = "./training-image-net-org/"
        
        dataset_name = ["mushroom", "cactus", "flower"]
        for name in dataset_name:
            merge( training_set_root + name + '/origin/*',  training_set_root + name + '/merge.txt')
        
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for name in dataset_name:
            output_directory =  directory + "/" + name + "/"
            store_raw_images(output_directory, training_set_root + name +'/merge.txt')

    else : print("Usage: download_images <dir_out>")






