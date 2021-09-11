import requests
import cv2
import json
import base64
import numpy as np
import io
from PIL import Image
from flask import Response

# url = "http://127.0.0.1:5000/api/adv"
url = "http://123.60.209.79:5000/api/adv/"

data = {
    "adv_level":3,
    "labels":[0,1],
    "imgs":["/test/upload/dataset_161451161238/jpg/LIDC-IDRI-0001/3000566.000000-03192/1-001_1624446023000.jpg","/test/upload/dataset_161451161238/jpg/LIDC-IDRI-0001/3000566.000000-03192/1-001_1624446023000.jpg"]
    }
headers = {'Content-Type': 'application/json;charset=UTF-8'}
response = requests.post(url,data=json.dumps(data),headers=headers)
print(response.text)