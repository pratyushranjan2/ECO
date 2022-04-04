from time import sleep
from PIL import Image
from feature_extractor import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
import pickle
import torch
import torch.nn as nn
import requests
from io import BytesIO
import cv2
from detection.cloth_detection import Detect_Clothes_and_Crop
from detection.utils_my import Read_Img_2_Tensor, Save_Image, Load_DeepFashion2_Yolov3

app = Flask(__name__)

#fe = FeatureExtractor()
#model = Load_DeepFashion2_Yolov3()
max_matches = 6
packets = []
uploaded_img_path = None

meta_hoodie = ['S M L', 'S M', 'M L', 'S M L XL', 'L', 'S M L']
meta_footwear = ['6 7 8 9 10', '8 9 10', '6 7 8', '8', '6 10', '11']
location = ['Phase-1 Electronic City Bangalore', 'HSR Layout Bangalore', 'Chandapura Bangalore', 'Kormangala Bangalore', 'BTM Layout Bangalore', 'Venkatapura Bangalore']
costs = ['\u20B9 300','\u20B9 400','\u20B9 350','\u20B9 450','\u20B9 250','\u20B9 300',]
rating = [4, 5, 4, 3, 4, 4]

with open('footwear_features.pkl', 'rb') as f:
    footwear_features = pickle.load(f)
with open('nike_men_hoodie_features.pkl', 'rb') as f:
    nike_men_hoodie_features = pickle.load(f)


def getScores(img_url=None, img_path=None):
    global uploaded_img_path
    global packets
    if img_url:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(img_path)
    
    uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + 'sample.jpg'
    img.save(uploaded_img_path)

    img_tensor = Read_Img_2_Tensor(uploaded_img_path)
    img_crop, detected = Detect_Clothes_and_Crop(img_tensor, model)
    
    if detected:
        features = nike_men_hoodie_features
        Save_Image(img_crop, uploaded_img_path)
        hoodie = True
    else:
        features = footwear_features
        hoodie = False
    
    img = Image.open(uploaded_img_path)
    
    with torch.no_grad():
        target_vector = fe(img=img)[0] 
        scores = {}
        
        for path, vector in zip(features.keys(), features.values()):
            scores[path] = f'{nn.CosineSimilarity(dim=0)(target_vector, vector).item()*100: .2f}'

        scores = [(score,url) for url,score in sorted(scores.items(), key = lambda item: item[1], reverse=True)][0:max_matches]

        packets.clear()
        for i in range(len(scores)):
            packets.append({'url': scores[i][1],
                            'score': scores[i][0],
                            'sizes': meta_hoodie[i] if hoodie else meta_footwear[i],
                            'cost': costs[i],
                            'location': location[i],
                            'rating': rating[i]
                            })

@app.route('/rent', methods=['POST'])
def rent():
    path = request.form.get('path')
    print(f'rent request on {path}')
    sleep(1)
    #getScores(img_path=path)
    packets = []
    for i in range(2,6):
        packets.append({'url': f'../static/denim-data/denim{i}.jpeg', 'location': location[i-1]})
    return render_template('list.html',
                            packets = packets)

@app.route('/sort', methods=['POST'])
def sort():
    global uploaded_img_path
    global packets
    param = request.form.get('param')
    if param is not None:
        if param == 'cost':
            new_packets = sorted(packets, key = lambda d: d['cost'])
        elif param == 'location':
            new_packets = sorted(packets, key = lambda d: d['location'])
        elif param == 'score':
            new_packets = sorted(packets, key = lambda d: d['score'], reverse=True)
        else:
            print('sort param not identified')
            new_packets = packets
        return render_template('card2.html',
                            query_path = uploaded_img_path,
                            packets = new_packets)
    else:
        print(f'unable to receive sorting parameter')
        return render_template('card2.html')

@app.route('/extsearch', methods=['GET'])
def searchByUrl():
    global uploaded_img_path
    global packets
    url = request.args.get('url')
    if (url is not None):
        getScores(url)
        print(f'uploaded_img_path={uploaded_img_path}')
        return render_template('card2.html',
                               query_path=uploaded_img_path,
                               packets=packets)
    else:
        print('url not received')
        return render_template('card2.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    # if request.method == 'POST':
    #     vals = [request.form.get('query_img_url0'), request.form.get('query_img_url1'), request.form.get('query_img_url2')]
    #     if (vals[0] is not None):
    #         img_url = vals[0]
    #         hoodie = False
    #     elif (vals[1] is not None):
    #         img_url = vals[1]
    #         hoodie = False
    #     else:
    #         img_url = vals[2]
    #         hoodie = True

    #     getScores(img_url)
    #     meta = meta_hoodie if hoodie else meta_footwear

    #     return render_template('insta.html',
    #                            query_path=uploaded_img_path,
    #                            packets=packets
    #                            )

    # else:
    #     return render_template('card2.html')
    return render_template('insta.html')


if __name__=="__main__":
    app.run("0.0.0.0")
