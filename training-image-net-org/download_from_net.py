import urllib.request
import magic
import os, sys

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
    if len(sys.argv) > 1 :
        directory = sys.argv[1]
        urlfile = sys.argv[2]

        store_raw_images(directory, urlfile)

    else : print("Usage: download_images <dir_out> <urlfile>")


