import urllib.request
import magic
import os, sys

def store_raw_images(directory, indice, images_link):
    image_urls = urllib.request.urlopen(images_link).read().decode()
    pic_num = indice

    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in image_urls.split('\n'):
        try:
            print(str(pic_num) + " : " + i)
            address = urllib.request.urlopen(i, timeout=1)
            if(address != None):
                filename = directory+"/"+str(pic_num)+".jpg"
                urllib.request.urlretrieve(i, filename)
                if(magic.from_file( filename, mime=True ) == "image/jpeg"):
                    print("[PASS] Image save")
                    pic_num += 1
                else:
                    print("[FAIL] Image drop: not image/jpeg")

        except Exception as e:
            print("[FAIL] Exception")
            pass


if __name__ == '__main__':
    if len(sys.argv) > 1 :
        directory = sys.argv[1]
        indice = int(sys.argv[2])
        link = sys.argv[3]

        store_raw_images(directory, indice, link)

    else : print("Usage: download_images <dir_out> <first_indice> <link_http>")


