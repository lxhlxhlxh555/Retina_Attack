import requests
import json


url = "http://127.0.0.1:5000/api/test/"
# url = "http://123.60.209.79:5000/api/adv/"

data = {
    "type":"normal",
    "attack_types":["motion_blur","defocus_blur","rgb_shift"],
    "attack_levels":[5,5,5],
    "task":"classification",
    "dataset":"local",
    "img_dir":"./app/test_samples",
    "model":"./app/weights/jit_module_448_cpu.pth",
    }
headers = {'Content-Type': 'application/json;charset=UTF-8'}
response = requests.post(url,data=json.dumps(data),headers=headers)
print(response.text)
