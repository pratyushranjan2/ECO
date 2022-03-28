import numpy as np
import pandas as pd
import cv2
from PIL import Image
from detection.cloth_detection import Detect_Clothes_and_Crop
from detection.utils_my import Read_Img_2_Tensor, Save_Image, Load_DeepFashion2_Yolov3
from feature_extractor import FeatureExtractor
import requests
from io import BytesIO
import pickle

model = Load_DeepFashion2_Yolov3()
fe = FeatureExtractor()

urls = pd.read_csv('nike_men_hoodies.csv').figure
features = {}

for url in urls:
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img_path = 'temp.jpg'
    img.save(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = Read_Img_2_Tensor(img_path)

    # Clothes detection and crop the image
    img_crop = Detect_Clothes_and_Crop(img_tensor, model)
    Save_Image(img_crop, img_path)

    feature = fe(img = Image.open(img_path))[0]
    features[url] = feature

if len(features) == len(urls):
    with open('nike_men_hoodie_features.pkl', 'wb') as f:
        pickle.dump(features, f)
    print('features extracted successfully...')