import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from sklearn.cluster import KMeans
import pickle
import sys

CLUSTERS=300
VOCAB_DIR = "./vocab/"

def computa_descritores(img):
    return cv.xfeatures2d.SURF_create().detectAndCompute(img,None)[1]

def representa_histograma(img, kmeans):
    class_freq = {}
    desc = computa_descritores(img)
    pred_list = kmeans.predict(desc)
    for predicted in pred_list:
        if predicted not in class_freq.keys():
            class_freq[predicted] = 1
        else:
            class_freq[predicted] += 1
    
    freq_list = []
    for i in range(CLUSTERS):
        if i not in class_freq.keys():
            freq_list.append([0])
        else:
            freq_list.append([class_freq[i]])
    
    return class_freq, np.array(freq_list).astype(np.float32)


def busca(img, database, kmeans, max_results=5):
    img_freq = representa_histograma(img, kmeans)[1]
    
    results = []
    for filename in database:
        db_img = cv.imread(filename, 1)
        db_img_freq = representa_histograma(db_img, kmeans)[1]
        
        dist = cv.compareHist(img_freq, db_img_freq, cv.HISTCMP_CHISQR)
        
        results.append((filename, dist))
    
    
    return sorted(results, key=lambda x: x[1])[:max_results]


pastas = ["./vocab/faces/", 
          "./vocab/garfield/", 
          "./vocab/platypus/", 
          "./vocab/nautilus/", 
          "./vocab/elephant/", 
          "./vocab/gerenuk/"]

all_files = []
for pasta in pastas:
    files = os.listdir(pasta)
    for filename in files:
        all_files.append(pasta+filename)

with open('kmeans_obj.pkl', 'rb') as f:
        kmeans_obj2 = pickle.load(f)

img_input = cv.imread(sys.argv[1], 1)
results = busca(img_input, all_files, kmeans_obj2, 5)

print("Top " + str(len(results)) + " resultados:")

for i in results:
    print("Caminho do arquivo: " +str(i[0]) + ", distancia: " + str(i[1]))
    plt.figure()
    plt.imshow(cv.imread(i[0], 1))

plt.show()