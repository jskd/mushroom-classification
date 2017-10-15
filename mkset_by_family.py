from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.request import urlretrieve
import urllib, os, re, unidecode

TRAIN_PATH = "training_set/"
TEST_PATH = "test_set/"

def make_soup(url):
    html = urlopen(url).read()
    return BeautifulSoup(html, "html.parser")

def make_sets():

    family_file = open("family_list.txt", 'w')

    if not os.path.exists(TRAIN_PATH):
        os.makedirs(TRAIN_PATH)

    if not os.path.exists(TEST_PATH):
        os.makedirs(TEST_PATH)

    total_especes = 0
    total_images = 0

    for i in range(800):
        root = "http://mycorance.free.fr/valchamp/"
        current_page = "{}champi{}.htm".format(root, i)

        try:
            soup = make_soup(current_page)
            name = soup.find('h1').string.split(":")[-1].strip().upper()
            
            for elem in soup(text=re.compile(r'Famille')):
                famille = elem.parent.parent.text.replace("Famille : ", "")
                famille = unidecode.unidecode(famille)
                famille = re.sub('\(lyophyllacees pour certains auteurs\)', '', famille)
                famille = re.sub('tricholotomacees', 'tricholotomatacees', famille)
                famille = re.sub('tricholomatacees', 'tricholotomatacees', famille)
                famille = re.sub('sclerodermacees', 'sclerodermatacees', famille)
                famille = re.sub('phellinaceess', 'phellinacees', famille)
                famille = re.sub('lepiotacee', 'lepiotacees', famille)
                famille = re.sub('boletales', 'boletacees', famille)
                famille = re.sub('schizophyllacees_pro_', '', famille)
                famille = famille.replace(" ", "")
                famille_liste = famille.split(",")


            images = [img for img in soup.findAll('img')]
            images_links = []

            # Get images links
            for each in images:
                link = each.get('src')
                if not link.endswith(".gif") or link.endswith("pasdephoto.jpg"):
                    images_links += [link]

            for fam in famille_liste:
                if not os.path.exists(TRAIN_PATH + fam):
                    os.makedirs(TRAIN_PATH + fam)
                    family_file.write(fam+"\n")
                if not os.path.exists(TEST_PATH + fam):
                    os.makedirs(TEST_PATH + fam)

                for each in images_links:
                    filename = each.split('/')[-1]
                    current = "{}{}".format(root, each)
                    urlretrieve(current, "{}/{}/{}".format(TRAIN_PATH, fam, filename))
            
            print("# {}".format(name))
            total_especes += 1
            total_images += len(images_links)

        except Exception as e:
            pass

    family_file.close()
    print("Done. {} Species and {} images.".format(total_especes, total_images))

make_sets()