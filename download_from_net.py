import urllib.request
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
                urllib.request.urlretrieve(i, directory+"/"+str(pic_num)+".jpg")
                pic_num += 1
            
        except Exception as e:
            pass


if __name__ == '__main__':
    if len(sys.argv) > 1 :
        directory = sys.argv[1]
        indice = int(sys.argv[2])
        link = sys.argv[3]

        store_raw_images(directory, indice, link)

    else : print("Usage: download_images <dir_out> <first_indice> <link_http>")


