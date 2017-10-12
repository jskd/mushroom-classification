from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.request import urlretrieve
import urllib, os

def make_soup(url):
    html = urlopen(url).read()
    return BeautifulSoup(html, "html.parser")

def get_images():

    if not os.path.exists("training_set"):
        os.makedirs("training_set")
    
    os.chdir("training_set")

    total_especes = 0
    total_images = 0

    for i in range(1000):
        root = "http://mycorance.free.fr/valchamp/"
        current_page = "{}champi{}.htm".format(root, i)

        try:
            soup = make_soup(current_page)
            name = soup.find('h1').string.split(":")[-1].strip().upper()
            toxic = False

            images = [img for img in soup.findAll('img')]
            images_links = []

            for each in images:
                link = each.get('src')

                if ".gif" in link:
                    if "morttete.gif" in link or "attentio.gif" in link:
                        toxic = True
                else:
                    images_links += [link]
            
            for each in images_links:
                filename = each.split('/')[-1]
                current = "{}{}".format(root, each)
                urlretrieve(current, "{}_{}".format("T" if toxic else "G", filename))

            print("# {} ({})".format(name, "Toxic" if toxic else "Good"))
            total_especes += 1
            total_images += len(images_links)

        except Exception as e:
            pass

    print("Done. {} Species and {} images.".format(total_especes, total_images))

get_images()