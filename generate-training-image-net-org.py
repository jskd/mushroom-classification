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
        fo.write("".join( sorted(list(all_lines)) ))



def found_in_file(fname, search):
    logfile = open(fname, 'r')
    loglist = logfile.readlines()
    logfile.close()
    for line in loglist:
        if search in line:
            return True
    return False

def store_raw_images(directory, urlfile):
    if not os.path.exists(directory):
        os.makedirs(directory)


    good_log= directory+'/good.txt'
    bad_log= directory+'/bad.txt'

    fgw= open(directory+'/good.txt','a')
    fbw= open(directory+'/bad.txt','a')

    line = 1
    num_lines = sum(1 for line in open(urlfile))
    with open(urlfile) as f:
        for i in f:
            found= found_in_file(good_log, i) or found_in_file(bad_log, i)

            if not found:
                try:
                    info = "(line:" + str(line).zfill(5) + "/"+ str(num_lines).zfill(5) +", url:" + i.rstrip() + ")"
                    request = urllib.request.urlopen(i, timeout=1)
                    if request.getcode() != 200 :
                        fbw.write(i)
                        print("[FAIL] Reponse error " + info)
                    elif request.info().get_content_type() != 'image/jpeg' :
                        fbw.write(i)
                        print("[FAIL] Not jpeg      " + info)
                    else:
                        filename = directory+"/"+str(line)+".jpg"
                        with open(filename, 'wb') as fout:
                            fout.write(request.read())
                            fgw.write(i)
                            print("[PASS] Image save    " + info)

                except Exception as e:
                    fbw.write(i)
                    print("[FAIL] Exception     " + info)
                    pass
            else:
                print("[PASS] Déjà dl "+ str(line))
            line += 1
            urllib.request.urlcleanup()
    fgw.close()
    fbw.close()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        training_set_root = "./training-image-net-org/"

        dataset_name = ["mushroom","cactus","flower"]
        for name in dataset_name:
            merge( training_set_root + name + '/origin/*',  training_set_root + name + '/merge.txt')


        if not os.path.exists(directory):
            os.makedirs(directory)

        for name in dataset_name:
            output_directory =  directory + "/" + name + "/"
            store_raw_images(output_directory, training_set_root + name +'/merge.txt')

    else : print("Usage: generate-training-image-net-org <dir_out>")
