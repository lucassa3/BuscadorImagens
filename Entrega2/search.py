import numpy as np
import matplotlib.pyplot as plt
from keras_preprocessing import image
import keras.applications.mobilenetv2 as keras_app
import argparse
import pickle
import os
import re
import sys


IMAGES_DIR = "./images"
TERM_IDS_FILE = './term_ids.txt'

def mnetv2_input_from_image(img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return keras_app.preprocess_input(x)

def get_image_paths(image_dir):
    path_list = []
    for path, subdirs, files in os.walk(image_dir):
        for name in files:
            path_list.append(os.path.join(path, name))
    return path_list

def build_index(image_paths):
    set_idx = []

    mv2 = keras_app.MobileNetV2(input_shape=None, alpha=1.0, depth_multiplier=1, 
                                include_top=True, weights='imagenet', input_tensor=None, 
                                pooling=None, classes=1000)

    for img_path in image_paths:
        processed_img = mnetv2_input_from_image(image.load_img(img_path, target_size=(224,224)))
        pred = mv2.predict(processed_img)
        set_idx.append((img_path, pred))

    with open('set_idx.pkl', 'wb') as f:
        pickle.dump(set_idx, f)

def list_term_ids(txt_path):
    term_list = []
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            term_list.append(re.findall("'([^']*)'", line)[0])

    with open('term_list.pkl', 'wb') as f:
        pickle.dump(term_list, f)

def term_to_id(term, term_list, partial=False):   
    if not partial:
        if term in term_list:
            idx = term_list.index(term)
            return [idx]
        else:
            raise ValueError(f"term {term} not known by this search engine")
    else:
        indices = [i for i, s in enumerate(term_list) if term in s]
        
        if not indices:
            raise ValueError(f"partial term {term} not known by this search engine")

        return indices


def get_probabilities(term_id, set_idx):
    proba_list = []
    for i in range(0, len(set_idx)):
        for idx in term_id:
            proba_list.append((set_idx[i][0], set_idx[i][1][0][idx]))
    return proba_list


def get_top_probas(proba_list, top=5):
    sorted_probas = sorted(proba_list, key=lambda x: x[1], reverse=True)
    return sorted_probas[:top]

def plot_images_scores(proba_list):
    for img_path in proba_list:
        print(img_path[0] + " score: " + str(img_path[1]))
        plt.figure()
        plt.imshow(image.load_img(img_path[0], target_size=(224,224)))

    plt.show()

def build_engine():
    print("Building indexes, this may take some time...")
    image_paths = get_image_paths(IMAGES_DIR)
    build_index(image_paths)
    list_term_ids(TERM_IDS_FILE)


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--build", help="builds image set index to speed up search")
parser.add_argument("-s", "--search", help="enter searching term here")
parser.add_argument("-p", "--partial", help="you can search with partial terms")
args = parser.parse_args()

if args.build:
    build_engine()

term_list = []
set_idx = []

if not ('term_list.pkl' in os.listdir() and 'set_idx.pkl' in os.listdir()):
    build_engine()

with open('term_list.pkl', 'rb') as f:
    term_list = pickle.load(f)

with open('set_idx.pkl', 'rb') as f:
    set_idx = pickle.load(f)

term_id = 0
if args.partial:
    term_id = term_to_id(args.search, term_list, partial=True)
else:
    term_id = term_to_id(args.search, term_list, partial=False)

proba_list = get_probabilities(term_id, set_idx)
top_probas = get_top_probas(proba_list, top=5)
plot_images_scores(top_probas)