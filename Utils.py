import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

data_dir="./Data/flowers"
categories=["daisy", "dandelion", "rose", "sunflower", "tulip"]

data=[]

def make_data():
    for category in categories:
        path=os.path.join(data_dir, category)
        label=categories.index(category)

        for img in os.listdir(path):
            img_path=os.path.join(path, img)
            img_arr=cv2.imread(img_path)

            try:
                image=cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
                image=cv2.resize(image, (224, 224))

                image=np.array(image, dtype=np.float32)
                image=image / 255.0
                data.append([image, label])

            except Exception as e:
                pass

    print(len(data))

    with open("data.pickle", "wb") as pik:
        pickle.dump(data, pik)

make_data()

def load_data():
    with open("data.pickle", "rb") as pick:
        data=pickle.load(pick)

    np.random.shuffle(data)

    feature=[]
    labels=[]

    for img, label in data:
        feature.append(img)
        labels.append(label)

    feature=np.array(feature, dtype=np.float32)
    labels=np.array(labels)

    return [feature, labels]



