# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.keras.models import Model
# import numpy as np

# See https://keras.io/api/applications/ for details

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

# class FeatureExtractor:
#     def __init__(self):
#         base_model = VGG16(weights='imagenet')
#         self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

#     def extract(self, img):
#         """
#         Extract a deep feature from an input image
#         Args:
#             img: from PIL.Image.open(path) or tensorflow.keras.preprocessing.image.load_img(path)

#         Returns:
#             feature (np.ndarray): deep feature with the shape=(4096, )
#         """
#         img = img.resize((224, 224))  # VGG must take a 224x224 img as an input
#         img = img.convert('RGB')  # Make sure img is color
#         x = image.img_to_array(img)  # To np.array. Height x Width x Channel. dtype=float32
#         x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
#         x = preprocess_input(x)  # Subtracting avg values for each pixel
#         feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
#         return feature / np.linalg.norm(feature)  # Normalize

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.model = models.vgg16(pretrained=False)
        self.model.load_state_dict(torch.load('../vgg16_weights'))
        self.feature_extractor = nn.Sequential(
            self.model.features,
            self.model.avgpool,
            nn.Flatten(),
            self.model.classifier[0]
        )
        self.transform = transforms.Compose([
            transforms.Resize((224*2,224*2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
    def forward(self, path=None, img=None):
        if img is None:
            response = requests.get(path)
            img = Image.open(BytesIO(response.content))
        return self.feature_extractor(self.transform(img).unsqueeze(0))

# fe = FeatureExtractor()
# print(fe('https://m.media-amazon.com/images/I/81FuTPhYTxL._AC_SR255,340_.jpg'))