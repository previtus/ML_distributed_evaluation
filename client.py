import requests
from timeit import default_timer as timer

PORT = "5000"
YOLO_KERAS_REST_API_URL = "http://localhost:"+PORT+"/yolo_single_crop"

IMAGE_PATH = "small.jpg"
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# submit the request
start = timer()
r = requests.post(YOLO_KERAS_REST_API_URL, files=payload).json()
end = timer()
print("Time", (end-start))
print("request data", r)
