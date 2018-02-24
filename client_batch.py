import requests
from timeit import default_timer as timer

PORT = "5000"
YOLO_KERAS_REST_API_URL = "http://localhost:"+PORT+"/yolo_image_batch"

IMAGE_PATH = "small.jpg"

C = 2 # 8, 16

uids = []
payload = {}
for i in range(C):
    image = open(IMAGE_PATH, "rb").read()
    payload[str(42+i)] = image

# submit the request
start = timer()
r = requests.post(YOLO_KERAS_REST_API_URL, files=payload).json()
total_time = timer() - start
print("Time total", total_time, "divby"+str(C)+" =", total_time/float(C))
print("request data", r)
for i,item in enumerate(r['bboxes']):
    print(r['uids'][i]," = len bboxes", len(item), item)
    #for a in item:print(a)
