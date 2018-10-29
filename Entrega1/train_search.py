import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
from sklearn.cluster import KMeans
import pickle

CLUSTERS = 300

def computa_descritores(img):
    return cv.xfeatures2d.SURF_create().detectAndCompute(img,None)[1]

def le_descritores_imagens(pastas, max_itens=5):
    files_desc = []
    
    for pasta in pastas:
        files = os.listdir(pasta)[:max_itens if max_itens < len(os.listdir(pasta)) else len(os.listdir(pasta))]
        
        for filename in files:
            path = pasta + filename
            img = cv.imread(path, 1)
            desc = computa_descritores(img)
            files_desc.append((path, desc))
    
    return files_desc


def cria_vocabulario(descritores, sz=300):
    all_desc = descritores[0][1]
    
    for i in range(1, len(descritores)):
        all_desc = np.append(all_desc, descritores[i][1], 0)
    
    
    kmeans = KMeans(n_clusters=sz, random_state=0)
    kmeans.fit(all_desc)
    
    return kmeans.cluster_centers_, kmeans
   

pastas = ["./vocab/faces/", "./vocab/garfield/", "./vocab/platypus/", "./vocab/nautilus/", "./vocab/elephant/", "./vocab/gerenuk/"]

descriptors = le_descritores_imagens(pastas, max_itens=5)

vocab, kmeans_obj = cria_vocabulario(descriptors, sz=CLUSTERS)

with open('kmeans_obj.pkl', 'wb') as f:
    pickle.dump(kmeans_obj, f)