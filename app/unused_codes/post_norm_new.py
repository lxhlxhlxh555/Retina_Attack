import requests
import cv2
import json
import base64
import numpy as np
import io
from PIL import Image
from flask import Response

# url = "http://127.0.0.1:5000/api/norm/"
url = "http://123.60.209.79:5000/api/norm/"
img1 = '/test/upload/dataset_161451161238/jpg/LIDC-IDRI-0001/3000566.000000-03192/1-001_1624446023000.jpg'
img2 = '/test/upload/dataset_161451161238/jpg/LIDC-IDRI-0001/3000566.000000-03192/1-001_1624446023000.jpg'
headers = {'Content-Type': 'application/json;charset=UTF-8'}
data = {
    "attack_types":["motion_blur","defocus_blur","rgb_shift"],
    "attack_levels":[5,5,5],
    "imgs":["/test/upload/dataset_161451161238/jpg/LIDC-IDRI-0001/3000566.000000-03192/1-001_1624446023000.jpg","/test/upload/dataset_161451161238/jpg/LIDC-IDRI-0001/3000566.000000-03192/1-001_1624446023000.jpg"]
}
headers = {}
response = requests.post(url,data=json.dumps(data),headers=headers)
print(response.text)
# j = json.loads(response.content)
# print(len(j['imgs']))
# for i,code in enumerate(j['imgs']):
#     code = base64.b64decode(code)
#     tmp = np.frombuffer(code,np.uint8)
#     print(tmp.shape)
#     img = cv2.imdecode(tmp, cv2.IMREAD_COLOR)
#     print(img.shape)
#     cv2.imwrite('{}.jpg'.format(i),img)