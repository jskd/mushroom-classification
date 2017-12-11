
# NEURAL NETWORK - IMAGE CLASSIFIER PROJECT

## Authors

* Jérome Skoda
* Joaquim Lefranc

## Génération du dataset imagenet.org

Python3 generate-training-image-net-org.py [repertoire de destination]

## mkset_by_toxicity
Récupère les images du site de champignons dans le dossier training_test/ par toxicité


## mkset_by_family
Récupère les images du site de champignons dans le dossier training_test/ par famille

## download_from_net.py
Récupère les images à partir d'un fichier d'url provenant de http://image-net.org/


## clean_up.py
Permet de nettoyer les images provenant de image-net, en se basant sur les exemples du dossier uglies/


## image_classifier.py

Lance l'apprentissage du réseau et le test

## Paquets pip nécessaires :

* beautifulsoup4
* opencv-python
* tensorflow
* unidecode
* python-magic
* glob2

## Supression des images corrompue

* jpeginfo -d **/*.jpg
* jpeginfo -c **/*.jpg | grep ERROR
* A suppr à la main
* jpeginfo -c **/*.jpg | grep WARNING
* A suppr à la main
